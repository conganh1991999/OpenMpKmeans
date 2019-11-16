.KEEP_STATE:

all: seq omp cuda mpi

DFLAGS      =
OPTFLAGS    = -O -NDEBUG
OPTFLAGS    = -g -pg
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=1
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v
LDFLAGS     = $(OPTFLAGS)
LIBS        =

OMPFLAGS    = -fopenmp

CC          = clang
MPICC       = mpicc
NVCC        = nvcc

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = kmeans.h

#------ OpenMP version -----------------------------------------
OMP_SRC     = omp_main.c \
	      omp_kmeans.c

OMP_OBJ     = $(OMP_SRC:%.c=%.o)

omp_kmeans.o: omp_kmeans.c $(H_FILES)
	$(CC) $(CFLAGS) $(OMPFLAGS) -c omp_kmeans.c

omp: omp_main
omp_main: $(OMP_OBJ) file_io.o
	$(CC) $(LDFLAGS) $(OMPFLAGS) -o omp_main $(OMP_OBJ) file_io.o $(LIBS)

#------ sequential version -------------------------------------
SEQ_SRC     = seq_main.c   \
              seq_kmeans.c \
	      file_io.c    \
	      wtime.c

SEQ_OBJ     = $(SEQ_SRC:%.c=%.o)

$(SEQ_OBJ): $(H_FILES)

seq: seq_main
seq_main: $(SEQ_OBJ) $(H_FILES)
	$(CC) $(LDFLAGS) -o seq_main $(SEQ_OBJ) $(LIBS)

#----------------------------------------------------------------
clean:
	rm -rf *.o omp_main seq_main mpi_main cuda_main \
	       core* .make.state gmon.out     \
               *.cluster_centres *.membership \
               Image_data/*.cluster_centres   \
               Image_data/*.membership        \
               profiles/
