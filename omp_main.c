#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "kmeans.h"

#include <omp.h>

int _debug;

/* usage() */
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -p nproc       : number of threads (default system allocated)\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

/* main() */
int main(int argc, char **argv) {
           int    opt;
    extern char   *optarg;
    extern int    optind;
           int    i, j, nthreads;
           int    isBinaryFile, is_output_timing;
           int    numClusters, numCoords, numObjs;
           int    *membership;
           char   *filename;
           float  **objects;
           float  **clusters;
           float  threshold;
           double timing, io_timing, clustering_timing;

    // khởi tạo các giá trị mặc định
    _debug            = 0;
    nthreads          = 0;
    numClusters       = 0;
    threshold         = 0.001;
    numClusters       = 0;
    isBinaryFile      = 0;
    is_output_timing  = 0;
    filename          = NULL;

    while ((opt=getopt(argc,argv,"p:i:n:t:abdo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold = atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'p': nthreads = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case 'd': _debug = 1;
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }

    if (filename == 0 || numClusters <= 1) usage(argv[0], threshold);

    if (nthreads > 0)
        omp_set_num_threads(nthreads);

    if (is_output_timing) io_timing = omp_get_wtime();

    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);

    if (is_output_timing) {
        timing            = omp_get_wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }      

    // tính toán
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    clusters = omp_kmeans(objects, numCoords, numObjs, numClusters, threshold, membership);

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = omp_get_wtime();
        clustering_timing = timing - clustering_timing;
    }       

    file_write(filename, numClusters, numObjs, numCoords, clusters, membership);

    free(membership);
    free(clusters[0]);
    free(clusters);

    if (is_output_timing) {
        io_timing += omp_get_wtime() - timing;

        printf("Number of threads = %d\n", omp_get_max_threads());
        printf("Input file:     %s\n", filename);
        printf("numObjs       = %d\n", numObjs);
        printf("numCoords     = %d\n", numCoords);
        printf("numClusters   = %d\n", numClusters);
        printf("threshold     = %.4f\n", threshold);

        printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", clustering_timing);
    }
    
    return(0);
}