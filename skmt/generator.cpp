//
//  graph_generator.cpp
//  graph_library
//
//  Created by Elijah Afanasiev on 17.08.17.
//  Copyright В© 2017 MSU. All rights reserved.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <utility>

#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

#ifndef uint32_t
#define uint32_t int
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T rand_uniform_val(int _upper_border)
{
    return (_T)(rand() % _upper_border);
}

template <>
int rand_uniform_val(int _upper_border)
{
    return (int)(rand() % _upper_border);
}

template <>
float rand_uniform_val(int _upper_border)
{
    return (float)(rand() % _upper_border) / _upper_border;
}

template <>
double rand_uniform_val(int _upper_border)
{
    return (double)(rand() % _upper_border) / _upper_border;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool save_to_file(int *_src_ids, int *_dst_ids, float *_weights, int _vertices_count, long long _edges_count, string _file_name, bool _weighted)
{
    ofstream graph_file(_file_name.c_str(), ios::binary);
    if (!graph_file.is_open())
        return false;
    
    graph_file.write(reinterpret_cast<const char*>(&_vertices_count), sizeof(int));
    graph_file.write(reinterpret_cast<const char*>(&_edges_count), sizeof(long long));
    
    for (long long i = 0; i < _edges_count; i++)
    {
        graph_file.write(reinterpret_cast<const char*>(&_src_ids[i]), sizeof(int));
        graph_file.write(reinterpret_cast<const char*>(&_dst_ids[i]), sizeof(int));
        
        if(_weighted)
            graph_file.write(reinterpret_cast<const char*>(&_weights[i]), sizeof(float));
    }
    
    graph_file.close();
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void get_cmd_params(int _argc, char **_argv, int &_scale, int &_avg_degree, string &_file_name, string &_type, bool &_directed, bool &_weighted, int &_threads)
{
    // set deafualt params
    _scale = 14;
    _avg_degree = 32;
    _file_name = "graph.data";
    _type = "RMAT";
    _directed = true;
    _weighted = true;
    _threads = 1;
    
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if (option.compare("-s") == 0)
        {
            _scale = atoi(_argv[++i]);
        }
        
        if (option.compare("-e") == 0)
        {
            _avg_degree = atoi(_argv[++i]);
        }
        
        if (option.compare("-file") == 0)
        {
            _file_name = _argv[++i];
        }
        
        if (option.compare("-type") == 0)
        {
            _type = _argv[++i];
        }
        
        if (option.compare("-undirected") == 0)
        {
            _directed = false;
        }
        
        if (option.compare("-unweighted") == 0)
        {
            _weighted = false;
        }
        
        if (option.compare("-t") == 0)
        {
            _threads = atoi(_argv[++i]);
        }
    }

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SSCA2(int *_src_ids, int *_dst_ids, float *_weights, int _vertices_count, long _max_clique_size, bool _directed, bool _weighted, long long &_edges_count)
{
    uint32_t TotVertices;
    uint32_t* clusterSizes;
    uint32_t* firstVsInCluster;
    uint32_t estTotClusters, totClusters;
    
    uint32_t *startVertex, *endVertex;
    uint32_t numEdges;
    uint32_t numIntraClusterEdges, numInterClusterEdges;
    float* weights;
    float MinWeight, MaxWeight;
    uint32_t MaxCliqueSize;
    uint32_t MaxParallelEdges = 1;
    double ProbUnidirectional = 1.0;
    double ProbIntercliqueEdges = 0.1;
    uint32_t i_cluster, currCluster;
    uint32_t *startV, *endV, *d;
    uint32_t estNumEdges, edgeNum;
    
    uint32_t i, j, k, t, t1, t2, dsize;
    double p;
    uint32_t* permV;
    
    // initialize RNG
    
    MinWeight = 0.0;
    MaxWeight = 1.0;
    TotVertices = _vertices_count;
    
    // generate clusters
    MaxCliqueSize = _max_clique_size;
    estTotClusters = 1.25 * TotVertices / (MaxCliqueSize/2);
    clusterSizes = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));
    
    for(i = 0; i < estTotClusters; i++)
    {
        clusterSizes[i] = 1 + (rand_uniform_val<double>(10000.0) *MaxCliqueSize);
    }
    
    totClusters = 0;
    
    firstVsInCluster = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));
    
    firstVsInCluster[0] = 0;
    for (i=1; i<estTotClusters; i++)
    {
        firstVsInCluster[i] = firstVsInCluster[i-1] + clusterSizes[i-1];
        if (firstVsInCluster[i] > TotVertices-1)
            break;
    }
    
    totClusters = i;
    
    clusterSizes[totClusters-1] = TotVertices - firstVsInCluster[totClusters-1];
    
    // generate intra-cluster edges
    estNumEdges = (uint32_t) ((TotVertices * (double) MaxCliqueSize * (2-ProbUnidirectional)/2) +
                              (TotVertices * (double) ProbIntercliqueEdges/(1-ProbIntercliqueEdges))) * (1+MaxParallelEdges/2);
    
    if ((estNumEdges > ((1<<30) - 1)) && (sizeof(uint32_t*) < 8))
    {
        fprintf(stderr, "ERROR: long* should be 8 bytes for this problem size\n");
        fprintf(stderr, "\tPlease recompile the code in 64-bit mode\n");
        exit(-1);
    }
    
    edgeNum = 0;
    p = ProbUnidirectional;
    
    fprintf (stderr, "[allocating %3.3f GB memory ... ", (double) 2*estNumEdges*8/(1<<30));
    
    startV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));
    endV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));
    
    fprintf(stderr, "done] ");
    
    for (i_cluster=0; i_cluster < totClusters; i_cluster++)
    {
        for (i = 0; i < clusterSizes[i_cluster]; i++)
        {
            for (j = 0; j < i; j++)
            {
                for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1) * rand_uniform_val<double>(10000.0)); k++)
                {
                    startV[edgeNum] = j + \
                    firstVsInCluster[i_cluster];
                    endV[edgeNum] = i + \
                    firstVsInCluster[i_cluster];
                    edgeNum++;
                }
            }
            
        }
    }
    numIntraClusterEdges = edgeNum;
    
    //connect the clusters
    dsize = (uint32_t) (log((double)TotVertices)/log(2));
    d = (uint32_t *) malloc(dsize * sizeof(uint32_t));
    for (i = 0; i < dsize; i++) {
        d[i] = (uint32_t) pow(2, (double) i);
    }
    
    currCluster = 0;
    
    for (i = 0; i < TotVertices; i++)
    {
        p = ProbIntercliqueEdges;
        for (j = currCluster; j<totClusters; j++)
        {
            if ((i >= firstVsInCluster[j]) && (i < firstVsInCluster[j] + clusterSizes[j]))
            {
                currCluster = j;
                break;
            }
        }
        for (t = 1; t < dsize; t++)
        {
            j = (i + d[t] + (uint32_t)(rand_uniform_val<double>(10000.0) * (d[t] - d[t - 1]))) % TotVertices;
            if ((j<firstVsInCluster[currCluster]) || (j>=firstVsInCluster[currCluster] + clusterSizes[currCluster]))
            {
                for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1)* rand_uniform_val<double>(10000.0)); k++)
                {
                    if (p >  rand_uniform_val<double>(10000.0))
                    {
                        startV[edgeNum] = i;
                        endV[edgeNum] = j;
                        edgeNum++;
                    }
                }
            }
            p = p/2;
        }
    }
    
    numEdges = edgeNum;
    numInterClusterEdges = numEdges - numIntraClusterEdges;
    
    free(clusterSizes);
    free(firstVsInCluster);
    free(d);
    
    fprintf(stderr, "done\n");
    fprintf(stderr, "\tNo. of inter-cluster edges - %d\n", numInterClusterEdges);
    fprintf(stderr, "\tTotal no. of edges - %d\n", numEdges);
    
    // shuffle vertices to remove locality
    fprintf(stderr, "Shuffling vertices to remove locality ... ");
    fprintf(stderr, "[allocating %3.3f GB memory ... ", (double)(TotVertices + 2 * numEdges) * 8 / (1 << 30));
    
    permV = (uint32_t *)malloc(TotVertices*sizeof(uint32_t));
    startVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));
    endVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));
    
    for (i = 0; i<TotVertices; i++)
    {
        permV[i] = i;
    }
    
    for (i = 0; i<TotVertices; i++)
    {
        t1 = i + rand_uniform_val<double>(10000.0) * (TotVertices - i);
        if (t1 != i)
        {
            t2 = permV[t1];
            permV[t1] = permV[i];
            permV[i] = t2;
        }
    }
    
    for (i = 0; i<numEdges; i++)
    {
        startVertex[i] = permV[startV[i]];
        endVertex[i] = permV[endV[i]];
    }
    
    free(startV);
    free(endV);
    free(permV);
    
    // generate edge weights
    
    fprintf(stderr, "Generating edge weights ... ");
    weights = (float *)malloc(numEdges*sizeof(float));
    for (i = 0; i<numEdges; i++)
    {
        weights[i] = MinWeight + (float)(MaxWeight - MinWeight) * rand_uniform_val<float>(10000.0);
    }
    
    vector<vector<uint32_t> > dests(TotVertices);
    vector<vector<float> > weight_vect(TotVertices);
    
    _edges_count = numEdges;
    
    // add edges to graph
    for (uint32_t cur_edge = 0; cur_edge < numEdges; cur_edge++)
    {
        int from = startVertex[cur_edge];
        int to = endVertex[cur_edge];
        float edge_weight = weights[cur_edge];
        
        if(_directed)
        {
            _src_ids[cur_edge] = from;
            _dst_ids[cur_edge] = to;
            
            if(_weighted)
                _weights[cur_edge] = edge_weight;
        }
        
        if(!_directed)
        {
            _src_ids[cur_edge] = min(to, from);
            _dst_ids[cur_edge] = max(to, from);
            if(_weighted)
                _weights[cur_edge] = edge_weight;
        }

    }
    fprintf(stderr, "done\n");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void R_MAT(int *src_ids, int *dst_ids, float *weights, int _vertices_count, long _edges_count, int _a_prob,  int _b_prob,
           int _c_prob, int _d_prob, int _omp_threads,  bool _directed, bool _weighted)
{
    int n = (int)log2(_vertices_count);
    
    cout << "using " << _omp_threads << " threads" << endl;
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
                
                int step = (int)pow(2, n - (i + 1));
                
                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;
            
            int from = x_middle;
            int to = y_middle;
            float edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);
            
            if(_directed)
            {
                src_ids[cur_edge] = from;
                dst_ids[cur_edge] = to;
                
                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }
            
            if(!_directed)
            {
                src_ids[cur_edge] = min(to, from);
                dst_ids[cur_edge] = max(to, from);
                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void uniform_random(int *src_ids, int *dst_ids, float *weights, int _vertices_count, long _edges_count, int _omp_threads,  bool _directed, bool _weighted)
{
    int n = (int)log2(_vertices_count);
    
    cout << "using " << _omp_threads << " threads" << endl;
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++)
        {
            int from = rand() % _vertices_count;
            int to = rand() % _vertices_count;
            float edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);
            
            if(_directed)
            {
                src_ids[cur_edge] = from;
                dst_ids[cur_edge] = to;
                
                if(_weighted)
                weights[cur_edge] = edge_weight;
            }
            
            if(!_directed)
            {
                src_ids[cur_edge] = min(to, from);
                dst_ids[cur_edge] = max(to, from);
                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void convertToCSR(int* &src_ids, int* &dst_ids, float *weights, int _vertices_count, long _edges_count,  bool _directed, bool _weighted, string filePath)
{
    if(!_weighted)
    {
        cout << "I convert only weighted  graphs" << endl;
        return;
    }

    map<int, vector<pair<int, float>>> dict;

    for(long i = 0; i < _vertices_count; i++)
        dict.emplace(i, vector<pair<int, float>>());
    
    for(long i = 0; i < _edges_count; i++)
        dict[src_ids[i]].push_back(pair<int, float>(dst_ids[i], weights[i]));

    delete[] src_ids;
    delete[] dst_ids;
    delete[] weights;

    int* IA, *JA;
    float *A;
    IA = new int[_vertices_count + 1];
    JA = new int[_edges_count];
    A = new float[_edges_count];
    IA[0] = 0;
    long cntJA = 0;
    for(long i = 0; i < _vertices_count; i++)
    {
        size_t size = dict[i].size();
        // sort(dict[i].begin(), dict[i].end(), [](pair<int, float> a, pair<int, float> b) {return a.first < b.first;});
        IA[i + 1] = IA[i] + size;
        for(long j = 0; j < size; j++)
        {
            JA[cntJA] = dict[i][j].first;
            A[cntJA] = dict[i][j].second;
            cntJA++;
        }
    }

    ofstream graph_file(filePath, ios::binary);
    
    uint64_t version = 1;
    uint64_t sizeEdgeTy = 4;
    uint64_t nvtxs = _vertices_count;
    uint64_t nedges = _edges_count;

    graph_file.write(reinterpret_cast<const char*>(&version), sizeof(uint64_t));
    graph_file.write(reinterpret_cast<const char*>(&sizeEdgeTy), sizeof(uint64_t));

    graph_file.write(reinterpret_cast<const char*>(&nvtxs), sizeof(uint64_t));
    graph_file.write(reinterpret_cast<const char*>(&nedges), sizeof(uint64_t));
    
    for(int i = 1; i < _vertices_count + 1; i++)
    {
        uint64_t tmp = IA[i];
        graph_file.write(reinterpret_cast<const char*>(&tmp), sizeof(uint64_t));
    }

    for(int i = 0; i < _edges_count; i++)
    {
        uint32_t tmp = JA[i];
        graph_file.write(reinterpret_cast<const char*>(&tmp), sizeof(uint32_t));
    }

    for(int i = 0; i < _edges_count; i++)
    {
        uint32_t tmp = A[i] * 1000.0;
        graph_file.write(reinterpret_cast<const char*>(&tmp), sizeof(uint32_t));
    }
    
    graph_file.close();


    // for (long i = 0; i < _edges_count; i++)
    //     cout << A[i] << " ";
    // cout << endl;

    // for (long i = 0; i < _vertices_count + 1; i++)
    //     cout << IA[i] << " ";
    // cout << endl;
    
    // for (long i = 0; i < _edges_count; i++)
    //     cout << JA[i] << " ";
    // cout << endl;

    delete[] IA;
    delete[] JA;
    delete[] A;

    return;
}

int main(int argc, char **argv)
{
    try
    {
        int scale, avg_degree;
        string file_name;
        int threads = omp_get_max_threads();
        bool weighted, directed;
        string type;
        
        get_cmd_params(argc, argv, scale, avg_degree, file_name, type, directed, weighted, threads);
        
        int vertices_count = pow(2.0, scale);
        long long edges_count = avg_degree * (long long)vertices_count;
        
        cout << "Using " << threads << " threads" << endl << endl;
        
        int *src_ids = new int[edges_count];
        int *dst_ids = new int[edges_count];
        float *weights;
        
        if(weighted)
        {
           weights = new float[edges_count];
        }
        
        // generate edges list graph in parallel
        double t1 = omp_get_wtime();
        
        if(type == "RMAT")
        {
            cout << "Generating RMAT graph" << endl;
            R_MAT(src_ids, dst_ids, weights, vertices_count, edges_count, 45, 20, 20, 15, threads, directed, weighted);
        }
        else if(type == "SSCA2")
        {
            cout << "Generating SSCA2 graph" << endl;
            SSCA2(src_ids, dst_ids, weights, vertices_count, avg_degree, directed, weighted, edges_count);
        }
        else if(type == "UR")
        {
            cout << "Generating Uniform Random graph" << endl;
            // uniform_random(src_ids, dst_ids, weights, vertices_count, avg_degree, directed, weighted, edges_count);
            uniform_random(src_ids, dst_ids, weights, vertices_count, edges_count, threads, directed, weighted);
        }
        else
        {
            cout << "Error! Unknow graph type" << endl;
            delete[] src_ids;
            delete[] dst_ids;
            
            if(weighted)
            {
                delete[] weights;
            }
            return 1;
        }
        
        double t2 = omp_get_wtime();
        cout << "Generation time: " << t2 - t1 << " sec" << endl << endl;
        
        cout << "Saving graph to " << file_name << endl;
        t1 = omp_get_wtime();
        save_to_file(src_ids, dst_ids, weights, vertices_count, edges_count, file_name, weighted);
        t2 = omp_get_wtime();
        cout << "Save time: " << t2 - t1 << " sec" << endl << endl;
        
        cout << "Converting graph to CSR and saving to " << file_name + "CSR" << endl;
        t1 = omp_get_wtime();
        convertToCSR(src_ids, dst_ids, weights, vertices_count, edges_count,  directed, weighted, file_name + "CSR");
        t2 = omp_get_wtime();
        cout << "Converting time: " << t2 - t1 << " sec" << endl << endl;

        cout << "done!" << endl << endl;
    }
    catch (const char *error)
    {
        cout << error << endl;
        getchar();
        return 1;
    }
    catch (...)
    {
        cout << "unknown error" << endl;
    }
    
    cout << "press any key to exit..." << endl;
    //getchar();
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////