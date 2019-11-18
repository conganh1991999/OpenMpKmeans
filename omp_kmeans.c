#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "kmeans.h"

/* Hàm euclid_dist_2(): tính bình phương khoảng cách (euclide) giữa 2 điểm trong không gian n chiều */
__inline static
float euclid_dist_2(int numdims, float *coord1, float *coord2)
{
    // numdims: số chiều
    // coord1[numdims]: tọa độ điểm thứ nhất
    // coord2[numdims]: tọa độ điểm thứ hai
    int i;
    float result = 0.0;

    for (i=0; i<numdims; i++)
        result += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(result);
}

/* Hàm find_nearest_cluster(): với một điểm bất kỳ, tìm id của (tâm) cụm gần nó nhất */
__inline static
int find_nearest_cluster(int numClusters, int numCoords, float *object, float **clusters)
{
    // numClusters: số lượng cụm
    // numCoords: số lượng tọa độ của điểm (bằng với số chiều)
    // object[numCoords]: tọa độ điểm
    // clusters[numClusters][numCoords]: tọa độ các tâm cụm hiện tại
    int   index, i;
    float dist, min_dist;

    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        // không cần phải lấy căn
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/* kmeans_clustering() : trả về tọa độ của các tâm cụm (chính xác nhất có thể) */
float** omp_kmeans(float **objects, int numCoords, int numObjs, int numClusters, float threshold, int *membership)
{
    // <input> objects[numObjs][numCoords]: dữ liệu điểm
    // <input> numCoords: số lượng tọa độ của mỗi điểm (bằng với số chiều)
    // <input> numObjs: số lượng điểm trong tập dữ liệu điểm
    // <input> numClusters: số lượng cụm
    // <input> threshold: giá trị sàn của delta để tiếp tục phân cụm
    // <output> membership[numObjs]: danh sách id cụm ứng với mỗi điểm

    int i, j, k, index, loop=0;
    float **newClusters;    // [numClusters][numCoords]: tọa độ của các tâm cụm mới
    int *newClusterSize;    // [numClusters]: số lượng điểm trong mỗi cụm mới
    float delta;		    // tỉ lệ phần trăm các điểm thay đổi cụm của nó
    float **clusters;	    // <output>: [numClusters][numCoords]
    
    double timing;

    int nthreads;				// số lượng thread
    int **local_newClusterSize; // [nthreads][numClusters]: mỗi thread có 1 mảng một chiều riêng
    float ***local_newClusters;	// [nthreads][numClusters][numCoords]: mỗi thread có 1 mảng hai chiều riêng

    nthreads = omp_get_max_threads();

    // cấp phát mảng cho <output>
    clusters = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*) malloc(numClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    // chọn ra các tâm cụm ban đầu từ dữ liệu điểm
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];

    // khởi tạo mảng membership
    for (i=0; i<numObjs; i++) membership[i] = -1;

    // khởi tạo mảng newClusterSize
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    // khởi tạo mảng newClusters
    newClusters = (float**) malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*) calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    // khởi tạo mảng local_newClusterSize
    local_newClusterSize = (int**) malloc(nthreads * sizeof(int*));
    assert(local_newClusterSize != NULL);
    local_newClusterSize[0] = (int*) calloc(nthreads*numClusters,sizeof(int));
    assert(local_newClusterSize[0] != NULL);
    for (i=1; i<nthreads; i++)
        local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;

    // khởi tạo mảng local_newClusters
    local_newClusters = (float***) malloc(nthreads * sizeof(float**));
    assert(local_newClusters != NULL);
    local_newClusters[0] =(float**) malloc(nthreads * numClusters * sizeof(float*));
    assert(local_newClusters[0] != NULL);
    for (i=1; i<nthreads; i++)
        local_newClusters[i] = local_newClusters[i-1] + numClusters;
    for (i=0; i<nthreads; i++) {
        for (j=0; j<numClusters; j++) {
            local_newClusters[i][j] = (float*) calloc(numCoords,sizeof(float));
            assert(local_newClusters[i][j] != NULL);
        }
    }

    /*  mỗi thread tính các cụm mới sử dụng một không gian bộ nhớ riêng,
        sau đó thread 0 (master) sẽ gom tổng tất cả các kết quả lại  */
    if (_debug) timing = omp_get_wtime();
    do {
        delta = 0.0;

        #pragma omp parallel \
                shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
        {
            int tid = omp_get_thread_num();
            #pragma omp for \
                        private(i,j,index) \
                        firstprivate(numObjs,numClusters,numCoords) \
                        schedule(static,500) \
                        reduction(+:delta)
            for (i=0; i<numObjs; i++) {
                // find the array index of nestest cluster center
                index = find_nearest_cluster(numClusters, numCoords,objects[i], clusters);

                // if membership changes, increase delta by 1
                if (membership[i] != index) delta += 1.0;

                // assign the membership to object i
                membership[i] = index;

                // update new cluster centers : sum of all objects located within (average will be performed later)
                local_newClusterSize[tid][index]++;
                for (j=0; j<numCoords; j++)
                    local_newClusters[tid][index][j] += objects[i][j];
            }
        } // end of #pragma omp parallel

        // let the main thread perform the array reduction
        for (i=0; i<numClusters; i++) {
            for (j=0; j<nthreads; j++) {
                newClusterSize[i] += local_newClusterSize[j][i];
                local_newClusterSize[j][i] = 0.0;
                for (k=0; k<numCoords; k++) {
                    newClusters[i][k] += local_newClusters[j][i][k];
                    local_newClusters[j][i][k] = 0.0;
                }
            }
        }

        // average the sum and replace old cluster centers with newClusters
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   // set back to 0
            }
            newClusterSize[i] = 0;   // set back to 0
        }
            
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    if (_debug) {
        timing = omp_get_wtime() - timing;
        printf("nloops = %2d (T = %7.4f)",loop,timing);
    }

    free(local_newClusterSize[0]);
    free(local_newClusterSize);

    for (i=0; i<nthreads; i++)
        for (j=0; j<numClusters; j++)
            free(local_newClusters[i][j]);
    free(local_newClusters[0]);
    free(local_newClusters);

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}