
/*
 * Note: Data leakage in cross-validation was fixed in this file.
 * The training set for each fold now properly excludes the test fold rows.
 * Fix by Sermet Pekin, 16.09.2025
 */
/*
@author andrii dobroshynski
*/

#include <stdio.h>
#include <stdlib.h>
#include "eval.h"
#include "../utils/log.h"

void hyperparameter_search(double **data, struct dim *csv_dim)
{
    // Init the options for number of trees to: 10, 100, 1000.
    size_t n = 3;

    size_t *estimators = malloc(sizeof(size_t) * n);
    estimators[0] = 10;
    estimators[1] = 50;
    estimators[2] = 100;

    // Init the options for the max depth for a tree to: 3, 7, 10.
    size_t *max_depths = malloc(sizeof(size_t) * n);
    max_depths[0] = 3;
    max_depths[1] = 7;
    max_depths[2] = 10;

    // Defaults based on SKLearn's defaults / hand picked in order to compare performance
    // with the same parameters.
    size_t max_features = 3;
    size_t min_samples_leaf = 2;
    //rufino@ipb.pt: to avoid "warning: unused variable ‘max_depth’"
    //size_t max_depth = 7;

    // Number of folds for cross validation.
    size_t k_folds = 5;

    // Best params computed from running the hyperparameter search.
    size_t best_n_estimators = -1;
    double best_accuracy = -1;

    for (size_t i = 0; i < n; ++i)
    {
        size_t n_estimators = estimators[i]; /* Number of trees in the forest. */

        for (size_t j = 0; j < n; ++j)
        {
            size_t max_depth = max_depths[j];

            RandomForestParameters params = {
                .n_estimators = n_estimators,
                .max_depth = max_depth,
                .min_samples_leaf = min_samples_leaf,
                .max_features = max_features
            };

            log_if_level(0, "[hyperparameter search] testing params:\n  n_estimators: %ld\n  max_depth: %ld\n  min_samples_leaf: %ld\n  max_features: %ld\n",
                   params.n_estimators,
                   params.max_depth,
                   params.min_samples_leaf,
                   params.max_features);




            double cv_accuracy = cross_validate(data,
                                                &params,
                                                csv_dim,
                                                k_folds);

            log_if_level(0, "[hyperparameter search] cross validation accuracy: %f%% (%ld%%)\n",
                       (cv_accuracy * 100),
                       (long)(cv_accuracy * 100));

            // Update best accuracy and best parameters found so far from the hyperparameter search.
            if (cv_accuracy > best_accuracy)
            {
                best_accuracy = cv_accuracy;
                best_n_estimators = n_estimators;
            }
        }
    }

    // Free auxillary buffers.
    free(estimators);
    free(max_depths);

    printf("[hyperparameter search] run complete\n  best_accuracy: %f\n  best_n_estimators (trees): %ld\n",
           best_accuracy, best_n_estimators);
}

double eval_model(const DecisionTreeNode **random_forest,
                  double **data,
                  const RandomForestParameters *params,
                  const struct dim *csv_dim,
                  const ModelContext *ctx)
{
    // Keeping track of how many predictions have been correct. Accuracy can be
    // computed with 'num_correct' / 'rowsPerFold' (or how many predictions we make).
    long num_correct = 0;

    // Since we are evaluating the model on a single fold (to control overfitting), we start
    // iterating the rows for which we are getting predictions at an offset that can be computed
    // as 'testingFoldIdx * rowsPerFold' and make predictions for 'rowsPerFold' number of rows
    size_t row_id_offset = ctx->testingFoldIdx * ctx->rowsPerFold;
    for (size_t row_id = row_id_offset; row_id < row_id_offset + ctx->rowsPerFold; ++row_id)
    {
        int prediction = predict_model(&random_forest,
                                       params->n_estimators,
                                       data[row_id]);
        int ground_truth = (int)data[row_id][csv_dim->cols - 1];

        log_if_level(1, "majority vote:  %ld |  ground truth: %d\n",
                prediction, ground_truth);

        // if (log_level > 1)
        //     printf("majority vote: %d | %d ground truth\n", prediction, ground_truth);

        if (prediction == ground_truth)
            ++num_correct;
    }
    return (double)num_correct / (double)ctx->rowsPerFold;
}

double cross_validate(double **data,
                      const RandomForestParameters *params,
                      const struct dim *csv_dim,
                      const size_t k_folds)
                      //const int k_folds);
                      //rufino@ipb.pt: to avoid the following warning in a loop
                      //warning: comparison of integer expressions of different signedness: ‘size_t’ {aka ‘long unsigned int’} and ‘int’

{
    double sumAccuracy = 0;
    size_t rows = csv_dim->rows;
    size_t cols = csv_dim->cols;
    size_t rowsPerFold = rows / k_folds;

    for (size_t foldIdx = 0; foldIdx < k_folds; ++foldIdx)
    {
        // Allocate training data (all rows except the test fold)
        size_t train_rows = rows - rowsPerFold;
        double **train_data = malloc(train_rows * sizeof(double*));
        size_t train_idx = 0;
        size_t test_start = foldIdx * rowsPerFold;
        size_t test_end = test_start + rowsPerFold;
        for (size_t i = 0; i < rows; ++i) {
            if (i < test_start || i >= test_end) {
                train_data[train_idx++] = data[i];
            }
        }
        struct dim train_dim = {train_rows, cols};
        const ModelContext ctx = {
            .testingFoldIdx = foldIdx,
            .rowsPerFold = rowsPerFold
        };
        // Train on training data only
        const DecisionTreeNode **random_forest = train_model(
            train_data,
            params,
            &train_dim,
            &ctx);
        // Evaluate on the test fold (still using the full data array for eval_model, which uses ctx to select test rows)
        const double accuracy = eval_model(
            random_forest,
            data,
            params,
            csv_dim,
            &ctx);
        sumAccuracy += accuracy;
        free_random_forest(&random_forest, params->n_estimators);
        free(train_data);
    }
    return sumAccuracy / k_folds;
}
