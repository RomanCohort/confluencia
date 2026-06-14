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

#' Diagnose Python environment for Confluencia
#'
#' Checks if Python is available via reticulate, verifies the confluencia
#' package installation, and reports version information for Python,
#' confluencia, and key dependencies.
#'
#' @return A named list with diagnostic information:
#' \describe{
#'   \item{python_available}{Logical indicating if Python is available}
#'   \item{python_version}{Python version string, or NA if unavailable}
#'   \item{python_path}{Path to Python executable, or NA if unavailable}
#'   \item{confluencia_available}{Logical indicating if confluencia is installed}
#'   \item{confluencia_version}{Confluencia version, or NA if not installed}
#'   \item{numpy_available}{Logical indicating if NumPy is installed}
#'   \item{numpy_version}{NumPy version, or NA if not installed}
#'   \item{pandas_available}{Logical indicating if pandas is installed}
#'   \item{pandas_version}{Pandas version, or NA if not installed}
#'   \item{torch_available}{Logical indicating if PyTorch is installed}
#'   \item{torch_version}{PyTorch version, or NA if not installed}
#'   \item{viennarna_available}{Logical indicating if ViennaRNA is installed}
#'   \item{status}{Overall status: "OK", "WARNING", or "ERROR"}
#'   \item{message}{Summary message describing the diagnostic result}
#' }
#' @export
#' @examples
#' \dontrun{
#' diag <- cf_diagnose()
#' print(diag$status)
#' print(diag$message)
#' }
cf_diagnose <- function() {
  result <- list(
    python_available = FALSE,
    python_version = NA_character_,
    python_path = NA_character_,
    confluencia_available = FALSE,
    confluencia_version = NA_character_,
    numpy_available = FALSE,
    numpy_version = NA_character_,
    pandas_available = FALSE,
    pandas_version = NA_character_,
    torch_available = FALSE,
    torch_version = NA_character_,
    viennarna_available = FALSE,
    status = "ERROR",
    message = "Unknown error occurred"
  )

  # Check 1: Python availability
  py_available <- tryCatch({
    reticulate::py_available(initialize = TRUE)
  }, error = function(e) FALSE)

  if (!py_available) {
    result$status <- "ERROR"
    result$message <- "Python is not available. Install Python and ensure reticulate can find it."
    return(result)
  }

  result$python_available <- TRUE

  # Get Python version and path
  result$python_version <- tryCatch({
    as.character(reticulate::py_run_string(
      "import sys; print(sys.version.split()[0])"
    )$`__builtins__`$print)
  }, error = function(e) {
    tryCatch({
      reticulate::py_config()$version
    }, error = function(e2) NA_character_)
  })

  result$python_path <- tryCatch({
    reticulate::py_config()$python
  }, error = function(e) NA_character_)

  # Check 2: Confluencia package
  result$confluencia_available <- tryCatch({
    !is.null(reticulate::import("confluencia", convert = FALSE))
  }, error = function(e) FALSE)

  if (result$confluencia_available) {
    result$confluencia_version <- tryCatch({
      cf_mod <- reticulate::import("confluencia", convert = FALSE)
      if (!is.null(cf_mod$`__version__`)) {
        as.character(cf_mod$`__version__`)
      } else {
        NA_character_
      }
    }, error = function(e) NA_character_)
  }

  # Check 3: NumPy
  result$numpy_available <- tryCatch({
    !is.null(reticulate::import("numpy", convert = FALSE))
  }, error = function(e) FALSE)

  if (result$numpy_available) {
    result$numpy_version <- tryCatch({
      np <- reticulate::import("numpy", convert = FALSE)
      as.character(np$`__version__`)
    }, error = function(e) NA_character_)
  }

  # Check 4: Pandas
  result$pandas_available <- tryCatch({
    !is.null(reticulate::import("pandas", convert = FALSE))
  }, error = function(e) FALSE)

  if (result$pandas_available) {
    result$pandas_version <- tryCatch({
      pd <- reticulate::import("pandas", convert = FALSE)
      as.character(pd$`__version__`)
    }, error = function(e) NA_character_)
  }

  # Check 5: PyTorch
  result$torch_available <- tryCatch({
    !is.null(reticulate::import("torch", convert = FALSE))
  }, error = function(e) FALSE)

  if (result$torch_available) {
    result$torch_version <- tryCatch({
      torch <- reticulate::import("torch", convert = FALSE)
      as.character(torch$`__version__`)
    }, error = function(e) NA_character_)
  }

  # Check 6: ViennaRNA (RNA package)
  result$viennarna_available <- tryCatch({
    rna_mod <- reticulate::import("RNA", convert = FALSE)
    !is.null(rna_mod)
  }, error = function(e) FALSE)

  # Determine overall status
  issues <- character(0)

  if (!result$confluencia_available) {
    issues <- c(issues, "confluencia package not installed")
  }

  if (!result$numpy_available) {
    issues <- c(issues, "numpy not available")
  }

  if (!result$pandas_available) {
    issues <- c(issues, "pandas not available")
  }

  # ViennaRNA is optional but important for RNA structure prediction
  if (!result$viennarna_available) {
    # Don't count as error, just note it
    result$viennarna_note <- "ViennaRNA (RNA package) not available - RNA structure features will be disabled"
  }

  if (length(issues) == 0) {
    result$status <- "OK"
    result$message <- "All checks passed. Python environment is ready for confluencia."
    if (!result$viennarna_available) {
      result$message <- paste0(result$message, " Note: ViennaRNA not installed (optional).")
    }
  } else {
    result$status <- "ERROR"
    result$message <- paste("Issues found:", paste(issues, collapse = ", "))
  }

  # Ensure clean output order
  result <- result[c(
    "python_available", "python_version", "python_path",
    "confluencia_available", "confluencia_version",
    "numpy_available", "numpy_version",
    "pandas_available", "pandas_version",
    "torch_available", "torch_version",
    "viennarna_available",
    "status", "message"
  )]

  # Add viennarna_note if present
  if (!is.null(result$viennarna_note)) {
    result <- c(result, list(viennarna_note = result$viennarna_note))
    result$viennarna_note <- NULL
  }

  result
}