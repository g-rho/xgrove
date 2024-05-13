utils::globalVariables(c("left")) # resolves note on 'no visible binding for global variable 'left'' in group_by() in ln.123.

#' @importFrom gbm gbm
#' @importFrom gbm pretty.gbm.tree
#' @importFrom dplyr group_by
#' @importFrom dplyr summarise
#' @importFrom stats cor
#' @importFrom stats predict
#' @importFrom rpart rpart
#' @importFrom rpart rpart.control
#' @importFrom rpart.plot rpart.plot
#'
#' @title Explanation groves
#'
#' @description Compute surrogate groves to explain predictive machine learning model and analyze complexity vs. explanatory power.
#'
#' @details A surrogate grove is trained via gradient boosting using \code{\link[gbm]{gbm}} on \code{data} with the predictions of using of the \code{model} as target variable.
#' Note that \code{data} must not contain the original target variable! The boosting model is trained using stumps of depth 1.
#' The resulting interpretation is extracted from \code{\link[gbm]{pretty.gbm.tree}}. 
#' The column \code{upper_bound_left} of the \code{rules} and the \code{groves} value of the output object contains 
#' the split point for numeric variables denoting the uppoer bound of the left branch. Correspondingly, the 
#' \code{levels_left} column contains the levels of factor variables assigned to the left branch. 
#' The rule weights of the branches are given in the rightmost columns. The prediction of the grove is 
#' obtained as the sum of the assigned weights over all rows.       
#'
#' @param model   A model with corresponding predict function that returns numeric values.
#' @param data    Data that must not (!) contain the target variable.
#' @param ntrees  Sequence of integers: number of boosting trees for rule extraction.
#' @param pfun    Optional predict function \code{function(model, data)} returning a real number. Default is the \code{predict()} method of the \code{model}.
#' @param shrink  Sets the \code{shrinkage} argument for the internal call of \code{\link[gbm]{gbm}}. As the \code{model} usually has a deterministic response 
#' the default is 1 different to the default of \code{\link[gbm]{gbm}} applied train a model based on data.
#' @param b.frac  Sets the \code{bag.fraction} argument for the internal call of \code{\link[gbm]{gbm}}. As the \code{model} usually has a deterministic response 
#' the default is 1 different to the default of \code{\link[gbm]{gbm}} applied train a model based on data.
#' @param seed    Seed for the random number generator to ensure reproducible results (e.g. for the default \code{bag.fraction} < 1 in boosting).
#' @param ...     Further arguments to be passed to \code{gbm} or the \code{predict()} method of the \code{model}.
#'
#' @return List of the results:
#' @return \item{explanation}{Matrix containing tree sizes, rules, explainability \eqn{{\Upsilon}} and the correlation between the predictions of the explanation and the true model.}
#' @return \item{rules}{Summary of the explanation grove: Rules with identical splits are aggegated. For numeric variables any splits are merged if they lead to identical parititions of the training data.}
#' @return \item{groves}{Rules of the explanation grove.}
#' @return \item{model}{\code{gbm} model.}
#'
#' @export
#'
#' @examples
#' library(randomForest)
#' library(pdp)
#' data(boston)
#' set.seed(42)
#' rf <- randomForest(cmedv ~ ., data = boston)
#' data <- boston[,-3] # remove target variable
#' ntrees <- c(4,8,16,32,64,128)
#' xg <- xgrove(rf, data, ntrees)
#' xg
#' plot(xg)
#' 
#' # Example of a classification problem using the iris data.
#' # A predict function has to be defined, here for the posterior probabilities of the class Virginica.  
#' data(iris)
#' set.seed(42)
#' rf    <- randomForest(Species ~ ., data = iris)
#' data  <- iris[,-5] # remove target variable
#' 
#' pf <- function(model, data){
#'   predict(model, data, type = "prob")[,3]
#'   }
#'   
#' xgrove(rf, data, pfun = pf)
#'
#' @author \email{gero.szepannek@@web.de}
#'
#' @references \itemize{
#'     \item {Szepannek, G. and von Holt, B.H. (2023): Can’t see the forest for the trees -- analyzing groves to explain random forests,
#'            Behaviormetrika, DOI: 10.1007/s41237-023-00205-2}.
#'     \item {Szepannek, G. and Luebke, K.(2023): How much do we see? On the explainability of partial dependence plots for credit risk scoring,
#'            Argumenta Oeconomica 50, DOI: 10.15611/aoe.2023.1.07}.
#'   }
#'
#' @rdname xgrove
xgrove <- function(model, data, ntrees = c(4,8,16,32,64,128), pfun = NULL, shrink = 1, b.frac = 1, seed = 42, ...){
  
  set.seed(seed)
  if(is.null(pfun)) {
    surrogatetarget <- predict(model, data)
    if(!is.numeric(surrogatetarget) | !is.vector(surrogatetarget)) stop("Default predict method does not return a numeric vector. Please specify pfun argument!")
  }
  if(!is.null(pfun)){
    surrogatetarget <- pfun(model = model, data= data)
    if(!is.numeric(surrogatetarget) | !is.vector(surrogatetarget)) stop("pfun does not return a numeric vector!")
  }

  # compute surrogate grove for specified maximal number of trees
  data$surrogatetarget <- surrogatetarget
  surrogate_grove <- gbm::gbm(surrogatetarget ~., data = data, distribution = "gaussian", n.trees = max(ntrees), shrinkage = shrink, bag.fraction = b.frac, ...)
  if(surrogate_grove$interaction.depth > 1) stop("gbm interaction.depth is supposed to be 1. Please do not specify it differently within the ... argument.")

  # extract groves of different size and compute performance
  explanation     <- NULL
  groves          <- list()
  interpretation  <- list()
  for(nt in ntrees){
    predictions      <- predict(surrogate_grove, data, n.trees = nt, ...)

    rules <- NULL
    for(tid in 1:nt){
      tinf    <- gbm::pretty.gbm.tree(surrogate_grove, i.tree = tid)
      newrule <- tinf[tinf$SplitVar != -1,]
      newrule <- data.frame(newrule, pleft = tinf$Prediction[rownames(tinf) == newrule$LeftNode], pright = tinf$Prediction[rownames(tinf) == newrule$RightNode])
      rules   <- rbind(rules, newrule)
    }

    vars   <- NULL
    splits <- NULL
    csplits_left <- NULL
    pleft  <- NULL
    pright <- NULL

    for(i in 1:nrow(rules)){
      vars   <- c(vars,   names(data)[rules$SplitVar[i]+1])
      if(is.numeric(data[,rules$SplitVar[i]+1])){
        splits       <- c(splits, rules$SplitCodePred[i])
        csplits_left <- c(csplits_left, NA)
      }
      if(is.factor(data[,rules$SplitVar[i]+1])){
        levs <- levels(data[,(rules$SplitVar[i]+1)])
        lids <- surrogate_grove$c.splits[[(rules$SplitCodePred[i] +1)]] == -1
        if(sum(lids) == 1) levs <- levs[lids]
        if(sum(lids) > 1)  levs <- paste(levs[lids], sep = "|")
        csl <- levs[1]
        if(length(levs) > 1){for(j in 2:length(levs)) csl <- paste(csl, levs[j], sep = " | ")}
        splits       <- c(splits, "")
        csplits_left <- c(csplits_left, csl)
      }

      pleft  <- c(pleft,  rules$pleft[i])
      pright <- c(pright, rules$pright[i])
    }

    basepred <- surrogate_grove$initF
    df <- data.frame(vars, splits, left = csplits_left, pleft = round(pleft, 4), pright = round(pright,4))
    df <- dplyr::group_by(df, vars, splits, left)
    df_small <- as.data.frame(dplyr::summarise(df, pleft = sum(pleft), pright = sum(pright)))
    df <- as.data.frame(df)
    
    # merge rules for numeric variables 
    if(nrow(df_small) > 1){
      i <- 2
      while (i != 0){
        drop.rule <- FALSE  
        if(is.numeric(data[,df_small$vars[i]])){
          for(j in 1:(i-1)){
            if(df_small$vars[i] == df_small$vars[j]) {
              v1  <- data[,df_small$vars[i]] <= df_small$splits[i]
              v2  <- data[,df_small$vars[j]] <= df_small$splits[j]
              tab <- table(v1, v2)
              if(sum(diag(tab)) == sum(tab)) {
                df_small$pleft[j]  <- df_small$pleft[i] + df_small$pleft[j] 
                df_small$pright[j] <- df_small$pright[i] + df_small$pright[j] 
                drop.rule <- TRUE
              }
            }
          }
        }
        if(drop.rule) {df_small  <- df_small[-i,]}
        if(!drop.rule) {i <- i+1}
        if(i > nrow(df_small)) {i <- 0}
      }
    }
    
    # compute complexity and explainability statistics
    trees      <- nt
    rules      <- nrow(df_small) #
    ASE <- mean((data$surrogatetarget - predictions)^2)
    ASE0 <- mean((data$surrogatetarget - mean(data$surrogatetarget))^2)
    upsilon <- 1 - ASE / ASE0
    rho <- cor(data$surrogatetarget, predictions)

    df0      <- data.frame(vars = "Intercept", splits = NA, left = NA, pleft = basepred, pright = basepred)
    df       <- rbind(df0, df)
    df_small <- rbind(df0, df_small)

    # for better 
    colnames(df) <- colnames(df_small) <- c("variable", "upper_bound_left", "levels_left", "pleft", "pright")
    
    groves[[length(groves)+1]] <- df
    interpretation[[length(interpretation)+1]]   <- df_small
    explanation <- rbind(explanation, c(trees, rules, upsilon, rho))
  }
  names(groves) <- names(interpretation) <- ntrees
  colnames(explanation) <- c("trees","rules","upsilon","cor")

  res <- list(explanation = explanation, rules = interpretation, groves = groves, model = surrogate_grove)
  class(res) <- "xgrove"
  return(res)
}


#' @title Plot surrogate grove statistics
#'
#' @description Plot statistics of surrogate groves to analyze complexity vs. explanatory power.
#'
#' @param x    An object of class \code{xgrove}.
#' @param abs  Name of the measure to be plotted on the x-axis, either \code{"trees"}, \code{"rules"}, \code{"upsilon"} or \code{"cor"}.
#' @param ord  Name of the measure to be plotted on the y-axis, either \code{"trees"}, \code{"rules"}, \code{"upsilon"} or \code{"cor"}.
#' @param ...  Further arguments passed to \code{plot}.
#'
#' @return No return value.
#'
#' @examples
#' library(randomForest)
#' library(pdp)
#' data(boston)
#' set.seed(42)
#' rf <- randomForest(cmedv ~ ., data = boston)
#' data <- boston[,-3] # remove target variable
#' ntrees <- c(4,8,16,32,64,128)
#' xg <- xgrove(rf, data, ntrees)
#' xg
#' plot(xg)
#'
#' @author \email{gero.szepannek@@web.de}
#'
#' @rdname plot.xgrove
#' @export
plot.xgrove <- function(x, abs = "rules", ord = "upsilon", ...){
  i <- which(colnames(x$explanation) == abs)
  j <- which(colnames(x$explanation) == ord)
  plot(x$explanation[,i], x$explanation[,j], xlab = abs, ylab = ord, type = "b", ...)
}

#' @export
print.xgrove <- function(x, ...) print(x$explanation)


#' @title Explainability
#'
#' @description Compute explainability given predicted data of the model and an explainer.
#'
#' @param porig    An object of class \code{xgrove}.
#' @param pexp  Name of the measure to be plotted on the x-axis, either \code{"trees"}, \code{"rules"}, \code{"upsilon"} or \code{"cor"}.
#'
#' @return Numeric explainability upsilon.
#'
#' @references \itemize{
#'     \item {Szepannek, G. and Luebke, K.(2023): How much do we see? On the explainability of partial dependence plots for credit risk scoring,
#'            Argumenta Oeconomica 50, DOI: 10.15611/aoe.2023.1.07}.
#'   }
#'   
#' @examples
#' library(randomForest)
#' library(pdp)
#' data(boston)
#' set.seed(42)
#' # Compute original model
#' rf <- randomForest(cmedv ~ ., data = boston)
#' data <- boston[,-3] # remove target variable
#' # Compute predictions
#' porig <- predict(rf, data)
#'
#' # Compute surrogate grove
#' xg <- xgrove(rf, data)
#' pexp <- predict(xg$model, data, n.trees = 16)
#' upsilon(porig, pexp)
#'
#' @author \email{gero.szepannek@@web.de}
#'
#' @rdname upsilon
#' @export
upsilon <- function(porig, pexp){
  #porig   <- predict(model, data)
  #pexp    <- predict(explanation, data)
  ASE     <- mean((porig - pexp)^2)
  ASE0    <- mean((porig - mean(porig))^2)
  ups     <- 1 - ASE / ASE0
  return(ups)
}




#' @title Surrogate trees
#'
#' @description Compute surrogate trees of different depth to explain predictive machine learning model and analyze complexity vs. explanatory power.
#'
#' @details A surrogate grove is trained via gradient boosting using \code{\link[rpart]{rpart}} on \code{data} with the predictions of using of the \code{model} as target variable.
#' Note that \code{data} must not contain the original target variable! 
#'
#' @param model   A model with corresponding predict function that returns numeric values.
#' @param data    Data that must not (!) contain the target variable.
#' @param maxdeps Sequence of integers: Maximum depth of the trees.
#' @param cparam  Complexity parameter for growing the trees.
#' @param pfun    Optional predict function \code{function(model, data)} returning a real number. Default is the \code{predict()} method of the \code{model}.
#' @param ...     Further arguments to be passed to \code{\link[rpart]{rpart.control}} or the \code{predict()} method of the \code{model}.
#'
#' @return List of the results:
#' @return \item{explanation}{Matrix containing tree sizes, rules, explainability \eqn{{\Upsilon}} and the correlation between the predictions of the explanation and the true model.}
#' @return \item{rules}{List of rules for each tree.}
#' @return \item{model}{List of the \code{rpart} models.}
#'
#' @export
#'
#' @examples
#' library(randomForest)
#' library(pdp)
#' data(boston)
#' set.seed(42)
#' rf    <- randomForest(cmedv ~ ., data = boston)
#' data  <- boston[,-3] # remove target variable
#' maxds <- 1:7
#' st    <- sgtree(rf, data, maxds)
#' st
#' # rules for tree of depth 3
#' st$rules[["3"]]
#' # plot tree of depth 3
#' rpart.plot::rpart.plot(st$model[["3"]])
#'
#' @author \email{gero.szepannek@@web.de}
#'
#' @references \itemize{
#'     \item {Szepannek, G. and Laabs, B.H. (2023): Can’t see the forest for the trees -- analyzing groves to explain random forests,
#'            Behaviormetrika, submitted}.
#'     \item {Szepannek, G. and Luebke, K.(2023): How much do we see? On the explainability of partial dependence plots for credit risk scoring,
#'            Argumenta Oeconomica 50, DOI: 10.15611/aoe.2023.1.07}.
#'   }
#'
#' @rdname sgtree
sgtree <- function(model, data, maxdeps = 1:8, cparam = 0, pfun = NULL, ...){ #seed = 42

#browser()    
  #set.seed(seed)
  if(is.null(pfun)) {
    surrogatetarget <- predict(model, data)
    if(!is.numeric(surrogatetarget) | !is.vector(surrogatetarget)) stop("Default predict method does not return a numeric vector. Please specify pfun argument!")
  }
  if(!is.null(pfun)){
    surrogatetarget <- pfun(model = model, data= data)
    if(!is.numeric(surrogatetarget) | !is.vector(surrogatetarget)) stop("pfun does not return a numeric vector!")
  }
  
  # compute surrogate grove for specified maximal number of trees
  data$surrogatetarget <- surrogatetarget
  
  surrogate_trees <- list()
  rules  <- list()
  explanation     <- NULL   # #rules etc...
  
  for(md in maxdeps){
    surrogate_trees[[as.character(md)]] <- rpart::rpart(surrogatetarget ~ ., data = data, 
                                                        control = rpart::rpart.control(cp = cparam, minsplit = 1, minbucket = 1, maxcompete = 0, maxsurrogate = 0, maxdepth = md)) # ...
    if(is.null(surrogate_trees[[as.character(md)]]$splits)){
      rules[[as.character(md)]] <- NULL
      explanation <- rbind(explanation, c(1, 0, 0, 0))
    }
    if(!is.null(surrogate_trees[[as.character(md)]]$splits)){
      frame    <- surrogate_trees[[as.character(md)]]$frame 
      splits   <- surrogate_trees[[as.character(md)]]$splits
      csplit   <- surrogate_trees[[as.character(md)]]$csplit
      frame    <- frame[,c(1:3)] #[,c(1:3,5)]
      splits   <- splits[,c(1,2,4)]
      newrules    <- frame[frame$var != "<leaf>",]
      if(!is.matrix(splits)) {splits <- as.data.frame(t(splits))} # if only one split
      newrules    <- cbind(newrules, splits)
      newrules$left <- ifelse(newrules$ncat < 0, "<" , ">")
      newrules    <- newrules[,c(1,7,6,5,2,3)]   
      
      # For a factor, the index column contains the row number of the csplit matrix
      for(j in 1:nrow(newrules)){
        vn <- newrules$var[j]
        #if(is.factor(data[[vn]])){
        if(newrules$ncat[j] > 1){
          ind <- newrules$index[j]
          sel <- which(csplit[ind,] == 1)
          newrules$index[j] <- paste(levels(data[[vn]])[sel], collapse = " + ")
          newrules$left[j]  <- "among"
          # The columns record 1 if that level of the factor goes to the left, 3 if it goes to the right, 
          # and 2 if that level is not present at this node of the tree (or not defined for the factor).
        }
      }  
      rules[[as.character(md)]] <- newrules
      
      # compute complexity and explainability statistics
      trees      <- 1
      nrules      <- nrow(newrules)
      
      predictions <- predict(surrogate_trees[[as.character(md)]], data)
      ASE <- mean((data$surrogatetarget - predictions)^2)
      ASE0 <- mean((data$surrogatetarget - mean(data$surrogatetarget))^2)
      upsilon <- 1 - ASE / ASE0
      rho <- cor(data$surrogatetarget, predictions)
      
      explanation <- rbind(explanation, c(trees, nrules, upsilon, rho))
    }
    
  }
  colnames(explanation) <- c("trees","rules","upsilon","cor")
  
  res <- list(explanation = explanation, rules = rules, model = surrogate_trees)
  class(res) <- "sgtree"
  return(res)
}

#' @title Plot surrogate tree statistics
#'
#' @description Plot statistics of surrogate trees to analyze complexity vs. explanatory power.
#'
#' @param x    An object of class \code{sgtree}.
#' @param abs  Name of the measure to be plotted on the x-axis, either \code{"trees"}, \code{"rules"}, \code{"upsilon"} or \code{"cor"}.
#' @param ord  Name of the measure to be plotted on the y-axis, either \code{"trees"}, \code{"rules"}, \code{"upsilon"} or \code{"cor"}.
#' @param ...  Further arguments passed to \code{plot}.
#'
#' @return No return value.
#'
#' @examples
#' library(randomForest)
#' library(pdp)
#' data(boston)
#' set.seed(42)
#' rf <- randomForest(cmedv ~ ., data = boston)
#' data <- boston[,-3] # remove target variable
#' ntrees <- c(4,8,16,32,64,128)
#' xg <- xgrove(rf, data, ntrees)
#' xg
#' plot(xg)
#'
#' @author \email{gero.szepannek@@web.de}
#'
#' @rdname plot.sgtree
#' @export
plot.sgtree <- function(x, abs = "rules", ord = "upsilon", ...){
  i <- which(colnames(x$explanation) == abs)
  j <- which(colnames(x$explanation) == ord)
  plot(x$explanation[,i], x$explanation[,j], xlab = abs, ylab = ord, type = "b", ...)
}

#' @export
print.sgtree <- function(x, ...) print(x$explanation)
