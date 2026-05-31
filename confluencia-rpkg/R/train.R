#' Train a drug model from CSV
#'
#' Trains a drug efficacy prediction model from a CSV file and saves
#' the model bundle as a .joblib file for later use with cf_drug_predict().
#'
#' @param csv_path Path to the training CSV file
#' @param smiles_col Column name containing SMILES strings (default: "smiles")
#' @param target_col Column name containing target values (default: "efficacy")
#' @param env_cols Optional list of environmental parameter column names
#' @param model_name Model type: "ridge", "rf", "gbr", "hgb", "mlp" (default: "gbr")
#' @param test_size Fraction of data to use for validation (default: 0.2)
#' @param random_state Random seed for reproducibility (default: 42)
#' @param save_path Path to save the .joblib bundle (default: alongside CSV)
#' @return Named list with metrics: mae, rmse, r2, n_train, n_val, bundle_path
#' @export
#' @examples
#' \dontrun{
#' # Train from a CSV file
#' result <- cf_drug_train("my_drug_data.csv", model_name = "ridge")
#' print(result$bundle_path)  # Path to saved model
#'
#' # Use the trained model for prediction
#' pred <- cf_drug_predict(result$bundle_path, "CC(=O)Oc1ccccc1C(=O)O")
#' }
cf_drug_train <- function(csv_path, smiles_col = "smiles", target_col = "efficacy",
                           env_cols = NULL, model_name = "gbr",
                           test_size = 0.2, random_state = 42,
                           save_path = NULL) {
  bridge <- .cf_bridge()
  result <- bridge$drug_train(
    csv_path = csv_path,
    smiles_col = smiles_col,
    target_col = target_col,
    env_cols = env_cols,
    model_name = model_name,
    test_size = as.numeric(test_size),
    random_state = as.integer(random_state),
    save_path = save_path
  )
  as.list(result)
}

#' Train an epitope model from CSV
#'
#' Trains an epitope binding prediction model from a CSV file and saves
#' the model bundle as a .joblib file for later use with cf_epitope_predict().
#'
#' @param csv_path Path to the training CSV file
#' @param sequence_col Column name containing peptide sequences (default: "sequence")
#' @param target_col Column name containing target values (default: "binding")
#' @param env_cols Optional list of environmental parameter column names
#' @param model_name Model type: "ridge", "rf", "hgb", "mlp" (default: "hgb")
#' @param test_size Fraction of data to use for validation (default: 0.2)
#' @param random_state Random seed for reproducibility (default: 42)
#' @param save_path Path to save the .joblib bundle (default: alongside CSV)
#' @return Named list with metrics: mae, rmse, r2, n_train, n_val, bundle_path
#' @export
#' @examples
#' \dontrun{
#' # Train from a CSV file
#' result <- cf_epitope_train("my_epitope_data.csv", model_name = "hgb")
#' print(result$bundle_path)  # Path to saved model
#'
#' # Use the trained model for prediction
#' pred <- cf_epitope_predict(result$bundle_path, "SLYNTVATL")
#' }
cf_epitope_train <- function(csv_path, sequence_col = "sequence", target_col = "binding",
                               env_cols = NULL, model_name = "hgb",
                               test_size = 0.2, random_state = 42,
                               save_path = NULL) {
  bridge <- .cf_bridge()
  result <- bridge$epitope_train(
    csv_path = csv_path,
    sequence_col = sequence_col,
    target_col = target_col,
    env_cols = env_cols,
    model_name = model_name,
    test_size = as.numeric(test_size),
    random_state = as.integer(random_state),
    save_path = save_path
  )
  as.list(result)
}
