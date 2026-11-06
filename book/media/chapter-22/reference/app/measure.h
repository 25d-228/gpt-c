#ifndef DL_MEASURE_H
#define DL_MEASURE_H

/* CLI entry points. Each invocation owns and destroys all of its state. */
int dl_compare_model(int argc, char **argv);
int dl_compare_training(int argc, char **argv);

#endif
