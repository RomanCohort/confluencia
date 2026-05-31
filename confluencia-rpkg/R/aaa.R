#' @import reticulate
#' @noRd
.onLoad <- function(libname, pkgname) {
  # Auto-configure Python on package load
  cf_find_python()
}

# Package-level Python bridge object (created lazily)
.bridge_env <- new.env(parent = emptyenv())
.bridge_env$bridge <- NULL