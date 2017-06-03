library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(extrafont)
loadfonts()

options(stringsAsFactors = F)
x1 = read.delim("~/Downloads/cnn.csv",sep=' ',header=F)
colnames(x1) <- c("n","fold","acc")
x1$Model = 'CNN'
x1 <- x1 %>% group_by(n,Model) %>% summarize(accuracy=mean(acc),std=sd(acc))


x2 = read.delim("~/Downloads/mlp.csv",sep=' ',header=F)
colnames(x2) <- c("n","fold","acc")
x2$Model = 'MLP'
x2 <- x2 %>% group_by(n,Model) %>% summarize(accuracy=mean(acc),std=sd(acc))


x3 = read.delim("~/Downloads/leekasso.csv",sep=' ',header=F)
colnames(x3) <- c("n","fold","acc")
x3$Model = 'Leekasso'
x3 <- x3 %>% group_by(n,Model) %>% summarize(accuracy=mean(acc),std=sd(acc))

x4 = read.delim("~/Downloads/mlp_leek.csv",sep=' ',header=F)
colnames(x4) <- c("n","fold","acc")
x4$Model = 'MLP (Leek)'
x4 <- x4 %>% group_by(n,Model) %>% summarize(accuracy=mean(acc),std=sd(acc))

x <- rbind(x1,x2,x3,x4)
x <- x %>% mutate(low=accuracy-std,high=min(accuracy+std,0.99999))

p1 <- ggplot(x,aes(x=n,y=accuracy,color=Model)) +
      geom_point() +
      geom_line() +
      geom_errorbar(aes(ymin=low,ymax=high),size=0.75,width=1) +
      ylim(c(0.5,1)) +
      labs(title="Performance on 0 vs. 1 MNIST",
           x="Training Sample Size",
           y = "Accuracy on Heldout Sample") +
      scale_color_ipsum() +
      theme_ipsum(plot_title_size=20,axis_text_size=12,axis_title_size=12)


p2 <- x %>% filter(Model != "MLP (Leek)") %>%
        ggplot(aes(x=n,y=accuracy,color=Model)) +
        geom_point() +
        geom_line() +
        geom_errorbar(aes(ymin=low,ymax=high),size=0.75,width=1) +
        ylim(c(0.9,1)) +
        labs(title="Performance on 0 vs. 1 MNIST",
             x="Training Sample Size",
             y = "Accuracy on Heldout Sample") +
        scale_color_ipsum() +
        theme_ipsum(plot_title_size=20,axis_text_size=12,axis_title_size=12)


p <- plot_grid(p1, p2,ncol=1)

ggsave(filename = "~/Dropbox (HMS)/beamandrew.github.io/images/deep_learning_works_post/cnn.png",
       p,
       width = 10, height = 7.5)


