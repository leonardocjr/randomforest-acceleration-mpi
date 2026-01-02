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
#include "eval/eval.h"
#include "utils/argparse.h"
#include "utils/data.h"
#include "utils/utils.h"
#include "utils/log.h"




int main(int argc, char **argv)
{
    struct arguments arguments;
    parse_args(argc, argv, &arguments);


    set_log_level(arguments.log_level);

    // Optionally set the random seed if a specific random seed was provided via an argument.

    //rufino@ipb.pt: added seed variable to keep track of seed used; used RAND_MAX as default argv seed
    //if (arguments.random_seed)
    //    srand(arguments.random_seed);
    //else
    //    srand((unsigned int)time(NULL));
    unsigned int seed;
    if (arguments.random_seed!=RAND_MAX) seed=arguments.random_seed;
    else seed=(unsigned int)time(NULL);
    srand(seed);

    // Read the csv file from args which must be parsed now.

    const char *file_name = arguments.args[0];
    if (!file_name) {
        printf("Usage: %s <CSV_FILE> [--num_rows N] [--num_cols N] [--log_level N] [--seed N]\n", argv[0]);
        return 1;
    }

    // If the values for rows and cols were provided as arguments, then use them for the
    // 'dim' struct, otherwise call 'parse_csv_dims()' to parse the csv file provided to
    // compute the size of the csv file.
    struct dim csv_dim;

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
    double *data = malloc(sizeof(double) * csv_dim.rows * csv_dim.cols);
    parse_csv(file_name, &data, csv_dim);

    // Compute a checksum of the data to verify that loaded correctly.
    log_if_level(1, "data checksum = %f\n", _1d_checksum(data, csv_dim.rows * csv_dim.cols));


    //rufino@ipb.pt: keep note of the default values
    //const int k_folds = 5 ;
    const int k_folds = 20 ;

    log_if_level(0, "using:\n  k_folds: %d\n", k_folds);

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


    if (log_level > 0)
        print_params(&params);

    // Pivot the csv file data into a two dimensional array.
    double **pivoted_data;
    pivot_data(data, csv_dim, &pivoted_data);

    log_if_level(1, "checksum of pivoted 2d array: %f\n", _2d_checksum(pivoted_data, csv_dim.rows, csv_dim.cols));
    
    // Start the clock for timing.
    clock_t begin_clock = clock();

    double cv_accuracy = cross_validate(pivoted_data, &params, &csv_dim, k_folds);
    printf("cross validation accuracy: %f%% (%ld%%)\n",
           (cv_accuracy * 100),
           (long)(cv_accuracy * 100));

    // Record and output the time taken to run.
    clock_t end_clock = clock();
    printf("(time taken: %fs)\n", (double)(end_clock - begin_clock) / CLOCKS_PER_SEC);

    // Free loaded csv file data.
    free(data);
    free(pivoted_data);
}
