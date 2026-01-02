/*
@author andrii dobroshynski
*/

#include "forest.h"
#include <mpi.h>

const DecisionTreeNode *train_model_tree(double **data,
                                         const RandomForestParameters *params,
                                         const struct dim *csv_dim,
                                         long *nodeId /* Ascending node ID generator */,
                                         const ModelContext *ctx)
{
    DecisionTreeNode *root = empty_node(nodeId);
    DecisionTreeDataSplit data_split = calculate_best_data_split(data,
                                                                 params->max_features,
                                                                 csv_dim->rows,
                                                                 csv_dim->cols,
                                                                 ctx);
    
                                                                 
    log_if_level(1, "calculated best split for the dataset in train_model_tree\n"
           "half1: %ld\nhalf2: %ld\nbest gini: %f\nbest value: %f\nbest index: %d\n",
           data_split.data[0].length,
           data_split.data[1].length,
           data_split.gini,
           data_split.value,
           data_split.index);


    populate_split_data(root, &data_split);

    // Start building the tree recursively.
    grow(root,
         params->max_depth,
         params->min_samples_leaf,
         params->max_features,
         1 /* Current depth. */,
         csv_dim->rows,
         csv_dim->cols,
         nodeId,
         ctx);

    // Free any temp memory.
    free(data_split.data);

    return root;
}

const DecisionTreeNode **train_model(double **data,
                                     const RandomForestParameters *params,
                                     const struct dim *csv_dim,
                                     const ModelContext *ctx)
{
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    
    int n_trees = params->n_estimators;
    
    // calcula quantas arvores cada processo vai construir
    int trees_per_process = n_trees / numtasks;
    int remainder = n_trees % numtasks;
    
    // ind de início e fim para este processo
    int start_tree = rank * trees_per_process + (rank < remainder ? rank : remainder);
    int end_tree = start_tree + trees_per_process + (rank < remainder ? 1 : 0);
    int local_n_trees = end_tree - start_tree;

    log_if_level(1, "Rank %d: building trees [%d, %d] (%d trees)\n", 
                 rank, start_tree, end_tree, local_n_trees);

    // Random forest model which is stored as a contigious list of pointers to DecisionTreeNode structs.
    const DecisionTreeNode **random_forest = (const DecisionTreeNode **)
        malloc(sizeof(DecisionTreeNode *) * local_n_trees); 

    // Node ID generator. We use this such that every node in the tree gets assigned a strictly
    // increasing ID for debugging.
    long nodeId = 0;

    // Populate the array with allocated memory for the random forest with pointers to individual decision
    // trees.
    for (int i = 0; i < local_n_trees; ++i)
    {
        int tree_id = start_tree + i;
    
        unsigned int tree_seed;
        MPI_Bcast(&tree_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            tree_seed = rand();
        }
        
        MPI_Bcast(&tree_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        srand(tree_seed + tree_id);
        
        log_if_level(2, "Rank %d: building global tree %d (local %d)\n", 
                     rank, tree_id, i);
        
        random_forest[i] = train_model_tree(data, params, csv_dim, &nodeId, ctx);
    }
    
    log_if_level(1, "Rank %d: completed construction of %d trees\n", rank, local_n_trees);
    
    return random_forest;
}

int predict_model(const DecisionTreeNode ***random_forest, size_t n_estimators, double *row)
{
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  
    // calcula quantas arvores cada processo vai construir
    int trees_per_process = n_estimators / numtasks;
    int remainder = n_estimators % numtasks;
    
    // ind de início e fim para este processo
    int start_tree = rank * trees_per_process + (rank < remainder ? rank : remainder);
    int end_tree = start_tree + trees_per_process + (rank < remainder ? 1 : 0);
    int local_n_trees = end_tree - start_tree;

    int zeroes = 0;
    int ones = 0;
    
    for (int i = 0; i < local_n_trees; ++i)
    {
        int prediction;
        make_prediction((*random_forest)[i] /* root of the tree */,
                        row,
                        &prediction);

        if (prediction == 0)
            zeroes++;
        else if (prediction == 1)
            ones++;
        else
        {
            printf("Error: currently only support binary classification, i.e. prediction values 0/1, got: %d\n",
                   prediction);
            exit(1);
        }
    }
    
    // combinar os votos de todos os processos
    int global_zeroes = 0;
    int global_ones = 0;
    
    MPI_Allreduce(&zeroes, &global_zeroes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ones, &global_ones, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (global_ones > global_zeroes)
        return 1;
    else
        return 0;
}

void free_random_forest(const DecisionTreeNode ***random_forest, const size_t length)
{
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // cada processo libera apenas suas arvores locais
    int trees_per_process = length / numtasks;
    int remainder = length % numtasks;
    
    // ind de início e fim para este processo
    int start_tree = rank * trees_per_process + (rank < remainder ? rank : remainder);
    int end_tree = start_tree + trees_per_process + (rank < remainder ? 1 : 0);
    int local_n_trees = end_tree - start_tree;

    long freeCount = 0;
    for (int idx = 0; idx < local_n_trees; ++idx)
    {
        // Recursively free this DecisionTree rooted at the current node.
        free_decision_tree_node((*random_forest)[idx], &freeCount);
    }
    // Free the actual array of pointers to the nodes.
    free(*random_forest);

    log_if_level(2, "Rank %d: total DecisionTreeNode free: %ld\n", rank, freeCount);
}

void print_params(const RandomForestParameters *params)
{
    printf("using RandomForestParameters:\n  n_estimators: %ld\n  max_depth: %ld\n  min_samples_leaf: %ld\n  max_features: %ld\n",
           params->n_estimators,
           params->max_depth,
           params->min_samples_leaf,
           params->max_features);
}
