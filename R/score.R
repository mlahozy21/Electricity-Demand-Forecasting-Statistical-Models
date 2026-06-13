rmse<-function(y, ychap, digits=0)
{
  return(round(sqrt(mean((y-ychap)^2,na.rm=TRUE)),digits=digits))
}

mape<-function(y,ychap, eps=1e-8)
{
  # Mean Absolute Percentage Error. Guards:
  #  - na.rm=TRUE so missing values don't propagate to NA;
  #  - division only over entries with |y| > eps (near-zero actuals would make
  #    the percentage error explode / divide by ~0); those rows are dropped.
  keep <- is.finite(y) & is.finite(ychap) & (abs(y) > eps)
  if (!any(keep)) return(NA_real_)
  return(round(100*mean(abs(y[keep]-ychap[keep])/abs(y[keep]), na.rm=TRUE), digits=2))
}


rmse.old<-function(residuals, digits=0)
{
  return(round(sqrt(mean((residuals)^2,na.rm=TRUE)),digits=digits))
}


absolute_loss <- function(y, yhat)
{
  mean(abs(y-yhat),na.rm=TRUE)
}

bias <- function(y, yhat)
{
  mean(y-yhat,na.rm=TRUE)
}




pinball_loss <- function(y, yhat_quant, quant, output.vect=FALSE)
{
  yhat_quant <- as.matrix(yhat_quant)
  pinball_loss <- 0
  nq <- ncol(yhat_quant)
  loss_q <- array(0, dim=nq)

  for (q in 1:nq) {
    loss_q[q] <- mean(((y-yhat_quant[,q]) * (quant[q]-(y<yhat