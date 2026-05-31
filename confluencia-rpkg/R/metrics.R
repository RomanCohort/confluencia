#' Compute regression metrics (MAE, RMSE, R2)
#'
#' @param y_true Numeric vector of true values
#' @param y_pred Numeric vector of predicted values
#' @param prefix Optional prefix for metric names
#' @return Named numeric vector with mae, rmse, r2
#' @export
#' @examples
#' cf_reg_metrics(c(1, 2, 3), c(1.1, 2.0, 2.9))
cf_reg_metrics <- function(y_true, y_pred, prefix = "") {
  bridge <- .cf_bridge()
  result <- bridge$reg_metrics(
    y_true = as.numeric(y_true),
    y_pred = as.numeric(y_pred),
    prefix = prefix
  )
  unlist(as.list(result))
}