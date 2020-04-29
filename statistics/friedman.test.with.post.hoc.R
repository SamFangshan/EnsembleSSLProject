# The following code is adapted from https://gist.github.com/jacksonpradolima/b40aa19325787898fc5ddc3155752d94

friedman.test.with.post.hoc <- function(data, alpha = 0.05)
{ 
  library("ggplot2")
  library("scmamp")
  library("pgirmess")
  pre.results <- friedmanTest(data)
  imanDavenport.result <- imanDavenportTest(data)
  
  if(pre.results$p.value < alpha){
    post.results <- NULL
    
    if(length(colnames(data)) > 9){
      post.results <- postHocTest(data=data, test="friedman", correct="shaffer")
    }else{
      post.results <- postHocTest(data=data, test="friedman", correct="bergmann")
    }
    
    bold <- post.results$corrected.pval < alpha
    bold[is.na(bold)] <- FALSE
    writeTabular(table=post.results$corrected.pval, format='f', bold=bold, hrule=0, vrule=0)
    
    write.csv(post.results$summary, "summary.csv", row.names = TRUE)
    write.csv(post.results$raw.pval, "raw.csv", row.names = TRUE)
    write.csv(post.results$corrected.pval, "corrected.csv", row.names = TRUE)

    friedman.mc <- friedmanmc(data.matrix(data))

    write.csv(friedman.mc, "multiple_comparison.csv", row.names = TRUE)

    plt <- plotPvalues(post.results$corrected.pval, alg.order=order(post.results$summary)) + labs(title=paste("Corrected p-values using Bergmann and Hommel procedure",sep="")) + xlab("Algorithm") + ylab("Algotithm") + scale_fill_gradientn("Corrected p-values" , colours = c("grey15" , "grey30"))
    list.to.return <- list(Friedman = pre.results, ImanDavenport = imanDavenport.result, PostHoc = post.results, FriedmanMC = friedman.mc, Plt = plt)
    return(list.to.return)
  }
  else{
    print("The results where not significant. There is no need for a post-hoc test.")
    list.to.return <- list(Friedman = pre.results, ImanDavenport = imanDavenport.result, PostHoc = NULL, FriedmanMC = NULL, Plt = NULL)
    return(list.to.return)
  }
}
