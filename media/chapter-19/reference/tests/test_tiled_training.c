#define _POSIX_C_SOURCE 200809L
#include "training_fixture.h"

#include <unistd.h>

static size_t model_differences = 0;
static double gradient_error = 0.0, update_error = 0.0;

static int close_value(double a, double b, double tolerance)
{
    return isfinite(a) && isfinite(b) && fabs(a-b) <= tolerance*(1.0+fabs(b));
}

static int model_case(size_t layers, size_t heads, DlBackend backend)
{
    int result=0;
    DlModel full={0}, tiled={0};
    DlTrainTape a={0}, b={0}, doubled={0}, exact={0}, rejected={0};
    DlModelConfig c={.vocab=17,.layers=layers,.width=16,.query_heads=4,
        .kv_heads=heads,.ffn_width=20,.epsilon=1e-5f,.init_std=0.125f,
        .rope_theta=10000,.seed=42};
    enum { B=2, T=5, M=B*T, R=3, S=2 };
    size_t ids[M],targets[M];
    for (size_t i=0;i<M;++i) { ids[i]=(i*3+2)%17; targets[i]=(i*5+1)%17; }
    OK(dl_model_create(&full,c));
    OK(dl_model_create(&tiled,c));
    OK(dl_train_tape_create(&a,&full,B,T,1024*1024));
    OK(dl_train_tape_create_tiled(&b,&tiled,B,T,R,S,1024*1024));
    OK(dl_train_tape_create_tiled(&doubled,&tiled,B,2*T,R,S,1024*1024));
    size_t bytes=0,twice=0,reference_bytes=0;
    OK(dl_train_tape_payload_bytes(&b,&bytes));
    OK(dl_train_tape_payload_bytes(&doubled,&twice));
    OK(dl_train_tape_payload_bytes(&a,&reference_bytes));
    size_t linear=(layers+1)*M*(9*16+3*heads*4+3*20)+2*M*17;
    size_t constant=layers*(R*S+R*4+2*R)+(2*R*S+R*4+2*S*4+R);
    CHECK(b.arena.count==linear);
    CHECK(a.arena.count==linear+2*(layers+1)*B*4*T*T);
    CHECK(bytes==linear*sizeof(float)+(3*layers*M*4+constant)*sizeof(double));
    CHECK(twice+constant*sizeof(double)==2*bytes);
    OK(dl_train_tape_create_tiled(&exact,&tiled,B,T,R,S,bytes));
    STATUS(dl_train_tape_create_tiled(&rejected,&tiled,B,T,R,S,bytes-1),DL_RANGE);
    CHECK(rejected.model==NULL && rejected.arena.data==NULL && rejected.tiled==NULL);
    STATUS(dl_train_tape_create_tiled(&rejected,&tiled,B,T,SIZE_MAX,S,SIZE_MAX),DL_OVERFLOW);
    CHECK(rejected.model==NULL && rejected.tiled==NULL);
    for (size_t l=0;l<layers;++l) {
        CHECK(b.blocks[l].attention.scores.base==NULL && b.blocks[l].attention.probs.base==NULL);
        if (l!=0) CHECK(b.tiled[l].data!=b.tiled[l-1].data);
    }
    CHECK(b.attention_scratch.scores.base==NULL && b.attention_scratch.probs.base==NULL);
    double fa=0,fb=0;
    OK(dl_model_forward(&full,&a,backend,ids,targets,M,&fa));
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    CHECK(close_value(fa,fb,2e-6));
    for (size_t i=0;i<M*17;++i)
        CHECK(close_value(a.logits.base[a.logits.offset+i],b.logits.base[b.logits.offset+i],3e-6));
    OK(dl_model_backward(&full,&a,DL_SET));
    OK(dl_model_backward(&tiled,&b,DL_SET));
    for (size_t p=0;p<full.param_count;++p)
        for (size_t i=0;i<full.params[p].grad.count;++i) {
            double x=full.params[p].grad.data[i],y=tiled.params[p].grad.data[i];
            gradient_error=fmax(gradient_error,fabs(x-y));
            CHECK(close_value(y,x,7e-6));
        }
    for (size_t i=0;i<M*16;++i)
        CHECK(close_value(a.embedding_grad.base[a.embedding_grad.offset+i],
                          b.embedding_grad.base[b.embedding_grad.offset+i],7e-6));
    STATUS(dl_model_backward(&tiled,&b,DL_SET),DL_STATE);
    /* Two positions in EACH unique parameter, including the tied table and
     * every block norm/projection. Preserve analytic values before changing. */
    for (size_t p=0;p<tiled.param_count;++p) {
        size_t coordinates[]={0,tiled.params[p].value.count-1};
        for (size_t j=0;j<2;++j) {
            size_t i=coordinates[j];
            float *v=&tiled.params[p].value.data[i],old=*v;
            float plus=old+0.002f,minus=old-0.002f;
            double analytic=tiled.params[p].grad.data[i],losses[2]={0};
            for (size_t side=0;side<2;++side) {
                OK(dl_model_mark_changed(&tiled));
                *v=side==0?plus:minus;
                OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&losses[side]));
            }
            OK(dl_model_mark_changed(&tiled)); *v=old;
            double numeric=(losses[0]-losses[1])/((double)plus-(double)minus);
            CHECK(close_value(numeric,analytic,4e-4));
            ++model_differences;
        }
    }
    /* Corrupt only block0's statistic: earlier block1 reverse work can have
     * written gradients. The model must invalidate the WHOLE result. */
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    b.tiled[0].data[b.tiled[0].row_capacity]=0;
    STATUS(dl_model_backward(&tiled,&b,DL_SET),DL_NUMERIC);
    CHECK(!tiled.gradients_valid && !b.ready);
    STATUS(dl_model_sgd(&tiled,0.01),DL_STATE);
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    OK(dl_model_backward(&tiled,&b,DL_SET));
    OK(dl_model_forward(&full,&a,backend,ids,targets,M,&fa));
    OK(dl_model_backward(&full,&a,DL_ADD));
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    OK(dl_model_backward(&tiled,&b,DL_ADD));
    for (size_t p=0;p<full.param_count;++p)
        for (size_t i=0;i<full.params[p].grad.count;++i)
            CHECK(close_value(full.params[p].grad.data[i],tiled.params[p].grad.data[i],1.4e-5));
    printf("tiled model: L=%zu Hkv=%zu backend=%d full=%zu tiled=%zu doubled=%zu bytes\n",
           layers,heads,(int)backend,reference_bytes,bytes,twice);
    result=1;
done:
    dl_train_tape_destroy(&rejected); dl_train_tape_destroy(&exact);
    dl_train_tape_destroy(&doubled); dl_train_tape_destroy(&b); dl_train_tape_destroy(&a);
    dl_model_destroy(&tiled); dl_model_destroy(&full);
    return result;
}

static DlStatus use_tiled(TrainingFixture *f, size_t qr, size_t kc)
{
    dl_train_tape_destroy(&f->tape);
    dl_train_tape_destroy(&f->evaluation);
    f->state.attention=DL_ATTENTION_TILED;
    f->state.query_tile=qr; f->state.key_tile=kc;
    DlStatus s=dl_train_tape_create_tiled(&f->tape,&f->model,1,8,qr,kc,1024*1024);
    if (s!=DL_OK) return s;
    return dl_train_tape_create_tiled(&f->evaluation,&f->model,1,8,qr,kc,1024*1024);
}

static int persistent_close(const TrainingFixture *a, const TrainingFixture *b)
{
    if (a->state.rng.state!=b->state.rng.state || a->state.tokens_seen!=b->state.tokens_seen ||
        a->optim.updates!=b->optim.updates) return 0;
    for (size_t p=0;p<a->model.param_count;++p)
        for (size_t i=0;i<a->model.params[p].value.count;++i) {
            double x=a->model.params[p].value.data[i],y=b->model.params[p].value.data[i];
            update_error=fmax(update_error,fabs(x-y));
            if (!close_value(x,y,3e-5)) return 0;
        }
    for (size_t i=0;i<2*a->optim.count;++i)
        if (!close_value(a->optim.moments[i],b->optim.moments[i],3e-6)) return 0;
    return 1;
}

static int training_resume(DlBackend backend, int fixed)
{
    int result=0,fd=-1;
    char path[]="/tmp/dl-tiled-XXXXXX";
    TrainingFixture full={0},tiled={0},restored={0};
    DlWeights weights={0};
    DlTrainStats a={0},b={0},r={0};
    OK(fixture_create(&full,backend,fixed));
    OK(fixture_create(&tiled,backend,fixed));
    OK(fixture_create(&restored,backend,fixed));
    OK(use_tiled(&tiled,3,2)); OK(use_tiled(&restored,3,2));
    /* A correct mode in the state paired with the wrong tape is rejected
     * before the trainer opens a gradient group or consumes random numbers. */
    STATUS(dl_train_update(&tiled.optim,&tiled.state,&tiled.training,&tiled.batch,
                           &full.tape,&b),DL_SHAPE);
    CHECK(tiled.optim.updates==0 && tiled.state.rng.state==2026);
    for (size_t i=0;i<4;++i) {
        OK(dl_train_update(&full.optim,&full.state,&full.training,&full.batch,&full.tape,&a));
        OK(dl_train_update(&tiled.optim,&tiled.state,&tiled.training,&tiled.batch,&tiled.tape,&b));
        CHECK(close_value(a.loss,b.loss,3e-6));
        CHECK(persistent_close(&full,&tiled));
    }
    fd=mkstemp(path); CHECK(fd!=-1); CHECK(close(fd)==0); fd=-1;
    OK(dl_checkpoint_save(path,&tiled.optim,&tiled.state,&tiled.training,&tiled.valid,1024*1024));
    /* Wrong strategy and wrong tile size cannot silently resume the run. */
    STATUS(dl_checkpoint_load(path,&full.optim,&full.state,&full.training,&full.valid,1024*1024),DL_SHAPE);
    restored.state.query_tile=4;
    STATUS(dl_checkpoint_load(path,&restored.optim,&restored.state,&restored.training,&restored.valid,1024*1024),DL_SHAPE);
    CHECK(restored.optim.updates==0 && restored.state.tokens_seen==0);
    restored.state.query_tile=3;
    OK(dl_checkpoint_load(path,&restored.optim,&restored.state,&restored.training,&restored.valid,1024*1024));
    CHECK(fixture_equal(&tiled,&restored,0));
    OK(dl_checkpoint_load_weights(path,&weights,tiled.model.config,1024*1024,1024*1024));
    for (size_t p=0;p<weights.param_count;++p) {
        DlMatrix w=weights.params[p].value;
        CHECK(memcmp(w.base+w.offset,tiled.model.params[p].value.data,w.rows*w.cols*sizeof(float))==0);
    }
    for (size_t i=0;i<3;++i) {
        OK(dl_train_update(&full.optim,&full.state,&full.training,&full.batch,&full.tape,&a));
        OK(dl_train_update(&tiled.optim,&tiled.state,&tiled.training,&tiled.batch,&tiled.tape,&b));
        OK(dl_train_update(&restored.optim,&restored.state,&restored.training,&restored.batch,&restored.tape,&r));
        CHECK(persistent_close(&full,&tiled));
        CHECK(b.loss==r.loss && fixture_equal(&tiled,&restored,1));
    }
    printf("tiled train/resume: backend=%d fixed=%d seven matched updates, three exact continuations\n",
           (int)backend,fixed);
    result=1;
done:
    if (fd!=-1) (void)close(fd);
    (void)unlink(path);
    dl_weights_destroy(&weights);
    fixture_destroy(&restored); fixture_destroy(&tiled); fixture_destroy(&full);
    return result;
}

static int storage_scale(void)
{
    int result=0;
    DlModel m={0}; DlTrainTape full={0},tiled={0};
    DlModelConfig c={.vocab=256,.layers=2,.width=32,.query_heads=4,.kv_heads=2,
        .ffn_width=64,.epsilon=1e-5f,.init_std=0.125f,.rope_theta=10000,.seed=42};
    OK(dl_model_create(&m,c));
    for (size_t time=32;time<=256;time*=2) {
        OK(dl_train_tape_create(&full,&m,1,time,64*1024*1024));
        OK(dl_train_tape_create_tiled(&tiled,&m,1,time,16,32,64*1024*1024));
        size_t f=0,t=0; OK(dl_train_tape_payload_bytes(&full,&f));
        OK(dl_train_tape_payload_bytes(&tiled,&t));
        printf("training payload: T=%zu materialized=%zu tiled=%zu\n",time,f,t);
        dl_train_tape_destroy(&tiled); dl_train_tape_destroy(&full);
    }
    result=1;
done:
    dl_train_tape_destroy(&tiled); dl_train_tape_destroy(&full); dl_model_destroy(&m);
    return result;
}

int main(void)
{
    const size_t heads[]={1,2,4};
    for (int backend=0;backend<2;++backend) {
        for (size_t h=0;h<3;++h)
            if (!model_case(h==0?1:2,heads[h],(DlBackend)backend)) return EXIT_FAILURE;
        for (int fixed=0;fixed<2;++fixed)
            if (!training_resume((DlBackend)backend,fixed)) return EXIT_FAILURE;
    }
    if (!storage_scale()) return EXIT_FAILURE;
    printf("tiled model differences=%zu max gradient disagreement=%.6g max weight disagreement=%.6g\n",
           model_differences,gradient_error,update_error);
    puts("all tiled training checks passed");
    return EXIT_SUCCESS;
}
