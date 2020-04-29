#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
source("statistics/friedman.test.with.post.hoc.R")

if (length(args) == 0) {
    stop("Input file name must be specified!", call.=FALSE)
}

df <- read.table(args[1], header = TRUE, sep = ",")
friedman.test.with.post.hoc(df)
