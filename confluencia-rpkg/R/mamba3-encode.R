#' Encode sequence with Mamba3Lite encoder
#'
#' @param sequence Amino acid or nucleotide sequence string
#' @return Named list of numeric vectors (summary, local_pool, meso_pool, global_pool)
#' @export
cf_mamba3_encode <- function(sequence) {
  bridge <- .cf_bridge()
  result <- bridge$mamba3_encode(sequence = sequence)
  lapply(result, unlist)
}