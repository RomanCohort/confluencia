#' Get or create the ConfluenciaBridge Python object
#'
#' @return A reticulate Python object (ConfluenciaBridge instance)
#' @noRd
.cf_bridge <- function() {
  if (is.null(.bridge_env$bridge)) {
    # Ensure the bridge script is on sys.path
    bridge_dir <- system.file("python", package = "confluencia")
    if (bridge_dir == "") {
      stop("confluencia bridge script not found in inst/python/")
    }
    reticulate::py_run_string(sprintf(
      "import sys; sys.path.insert(0, r'%s')", bridge_dir
    ))
    mod <- reticulate::import("confluencia_bridge")
    .bridge_env$bridge <- mod$ConfluenciaBridge()
  }
  .bridge_env$bridge
}

#' Specify Python interpreter for Confluencia
#'
#' Set the Python interpreter to use for the reticulate bridge.
#' Must be called before any other cf_* function.
#'
#' @param python_path Path to the Python executable
#' @export
cf_use_python <- function(python_path) {
  reticulate::use_python(python_path, required = TRUE)
  .bridge_env$bridge <- NULL  # reset bridge
  invisible(NULL)
}

#' Auto-detect Python with Confluencia installed
#'
#' Searches for a Python interpreter with the confluencia packages
#' available, in order of priority:
#' \enumerate{
#'   \item CONFLUENCIA_PYTHON environment variable
#'   \item .venv in project root
#'   \item conda environment "confluencia"
#'   \item reticulate default
#' }
#'
#' @return Character vector with the Python path found (invisibly)
#' @export
cf_find_python <- function() {
  # 1. Check env var
  env_python <- Sys.getenv("CONFLUENCIA_PYTHON", "")
  if (env_python != "" && file.exists(env_python)) {
    cf_use_python(env_python)
    return(invisible(env_python))
  }

  # 2. Check .venv in project root
  project_root <- Sys.getenv("CONFLUENCIA_ROOT", "")
  if (project_root == "") {
    # Try to find from bridge script location
    bridge_dir <- system.file("python", package = "confluencia")
    if (bridge_dir != "") {
      project_root <- dirname(dirname(bridge_dir))
    }
  }

  if (project_root != "") {
    venv_python <- file.path(project_root, ".venv",
                             if (.Platform$OS.type == "windows") "Scripts/python.exe" else "bin/python")
    if (file.exists(venv_python)) {
      cf_use_python(venv_python)
      return(invisible(venv_python))
    }
  }

  # 3. Try conda
  tryCatch({
    conda_py <- reticulate::conda_python("confluencia")
    if (file.exists(conda_py)) {
      cf_use_python(conda_py)
      return(invisible(conda_py))
    }
  }, error = function(e) NULL)

  # 4. Use reticulate default
  invisible(NULL)
}

#' Convert Python dict to R data.frame
#' @param py_dict A Python dict of lists (as returned by ctm_simulate)
#' @return data.frame
#' @noRd
.py_dict_to_df <- function(py_dict) {
  as.data.frame(py_dict, stringsAsFactors = FALSE)
}

#' Convert Python dict to R named list/vector
#' @param py_dict A Python dict
#' @return named list
#' @noRd
.py_dict_to_list <- function(py_dict) {
  result <- as.list(py_dict)
  # If all values are scalars, simplify to named vector
  if (all(vapply(result, function(x) length(x) == 1 && is.numeric(x), logical(1)))) {
    result <- unlist(result)
  }
  result
}

#' Validate and convert env_params
#' @param env_params Named list or NULL
#' @return Python dict or NULL
#' @noRd
.validate_env_params <- function(env_params) {
  if (is.null(env_params)) return(NULL)
  if (!is.list(env_params) || is.null(names(env_params))) {
    stop("env_params must be a named list")
  }
  env_params
}