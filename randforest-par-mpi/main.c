/*
 * Modifications by Sermet Pekin , 19.09.2025 :
 * - Improved CSV parsing to handle long lines.
 * - Added debug information for CSV parsing.
 * - Added log_if_level function for conditional logging based on log_level.
 * - The training set for each fold now properly excludes the test fold rows.
 * - Added real data for testing and evaluation.
 * Fix by Sermet Pekin, 16.09.2025
 */

/*
@author andrii dobroshynski
*/

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "eval/eval.h"
#include "utils/argparse.h"
#include "utils/data.h"
#include "utils/utils.h"
#include "utils/log.h"




int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    struct arguments arguments;
    unsigned int seed;
    
    // rank 0 faz parse e setup inicial
    if (rank == 0) {
        parse_args(argc, argv, &arguments);
        set_log_level(arguments.log_level);
        
        if (arguments.random_seed != RAND_MAX) 
            seed = arguments.random_seed;
        else 
            seed = (unsigned int)time(NULL);
    }
    
    // broadcast da seed para garantir mesmos numeros aleatorios
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    srand(seed);

    // broadcast log_level para workers conseguirem logar
    int log_level_val;
    if (rank == 0) {
        log_level_val = arguments.log_level;
    }
    
    MPI_Bcast(&log_level_val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        set_log_level(log_level_val);
    }

    // Read the csv file from args which must be parsed now.
    const char *file_name = NULL;
    
    if (rank == 0) {
        file_name = arguments.args[0];
        if (!file_name) {
            printf("Usage: %s <CSV_FILE> [--num_rows N] [--num_cols N] [--log_level N] [--seed N]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // If the values for rows and cols were provided as arguments, then use them for the
    // 'dim' struct, otherwise call 'parse_csv_dims()' to parse the csv file provided to
    // compute the size of the csv file.
    struct dim csv_dim;
    double *data = NULL;

    if (rank == 0) {
      if (arguments.rows && arguments.cols)
        csv_dim = (struct dim){.rows = arguments.rows, .cols = arguments.cols};
      else
        csv_dim = parse_csv_dims(file_name);


      //rufino@ipv.pt: removed indentation to avoid "warning: this ‘else’ clause does not guard"
      //rufino@ipv.pt: added seed used
      log_if_level(0, "using:\n  seed: %d\n  verbose log level: %d\n  rows: %ld, cols: %ld\nreading from csv file:\n  \"%s\"\n",
               seed,
               arguments.log_level,
               csv_dim.rows,
               csv_dim.cols,
               file_name);


      // Allocate memory for the data coming from the .csv and read in the data.
      data = malloc(sizeof(double) * csv_dim.rows * csv_dim.cols);
      parse_csv(file_name, &data, csv_dim);

      // Compute a checksum of the data to verify that loaded correctly.
      log_if_level(1, "data checksum = %f\n", _1d_checksum(data, csv_dim.rows * csv_dim.cols));
    }
    
    // broadcast dimensoes
    MPI_Bcast(&csv_dim.rows, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&csv_dim.cols, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    
    // workers alocam espaco
    if (rank != 0) {
        data = malloc(sizeof(double) * csv_dim.rows * csv_dim.cols);
    }
    
    // broadcast dados
    int n_elements = (int)(csv_dim.rows * csv_dim.cols);
    MPI_Bcast(data, n_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //rufino@ipb.pt: keep note of the default values
    //const int k_folds = 5 ;
    const int k_folds = 20 ;

    if (rank == 0) {
      log_if_level(0, "using:\n  k_folds: %d\n", k_folds);
    }

    // Example configuration for a random forest model.
        //rufino@ipb.pt: keep note of the default values
        //.n_estimators = 3 /* Number of trees in the random forest model. */,
        //.max_depth = 7 /* Maximum depth of a tree in the model. */,
        //.min_samples_leaf = 3,
        //.max_features = 3
    const RandomForestParameters params = {
        .n_estimators = 20 /* Number of trees in the random forest model. */,
        .max_depth = 7 /* Maximum depth of a tree in the model. */,
        .min_samples_leaf = 3,
        .max_features = 20
    };

    // Print random forest parameters.
    if (rank == 0 && log_level > 0) {
        print_params(&params);
    }

    // Pivot the csv file data into a two dimensional array.
    double **pivoted_data;
    pivot_data(data, csv_dim, &pivoted_data);

    if (rank == 0) {
      log_if_level(1, "checksum of pivoted 2d array: %f\n", _2d_checksum(pivoted_data, csv_dim.rows, csv_dim.cols));
    }
    
    // Start the clock for timing.
    double cv_accuracy;
    clock_t begin_clock, end_clock;
    
    if (rank == 0) {
      begin_clock = clock();
    }
    
    cv_accuracy = cross_validate(pivoted_data, &params, &csv_dim, k_folds);
    
    if (rank == 0) {
      end_clock = clock();
      printf("cross validation accuracy: %f%% (%ld%%)\n",
           (cv_accuracy * 100),
           (long)(cv_accuracy * 100));
      printf("(time taken: %fs)\n", (double)(end_clock - begin_clock) / CLOCKS_PER_SEC);
    }

    // Free loaded csv file data.
    free(data);
    free(pivoted_data);
    MPI_Finalize();
    return 0;
}
