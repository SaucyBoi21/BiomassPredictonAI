library(tidyverse)
library(dplyr)
library(readxl)
library(ggridges)
library(reshape2)
library(lubridate)
library(RColorBrewer)
library(corrplot)
library(GGally)

# Create box plots for each date comparing data distribution for each tray. 

raw_data <- read.csv("harvest_complete.csv")
raw_data$date <- dmy(raw_data$date)
raw_data$treatment <-as.factor(with(raw_data, ifelse(tray_id == '1', '280 ppm',
                                                     ifelse(tray_id == 2, '160 ppm', '80 ppm'))))

biomass_data<- raw_data %>% dplyr::select(1,7,8,9,25)
manual_data <- raw_data %>% dplyr::select(1,3,10:12,25)
colnames(manual_data)
# Convert the raw data to a new data set where measuring variable names are included in a column.
# melt creates variable and values columns. Strings inside measure.vars taken as variable factor for variable.  

manual_reshape <- melt(manual_data, 
                    measure.vars = c("length_mm","width_mm","height_mm"))


manual_reshape %>% ggplot(aes(variable,value,color = treatment)) +
  geom_boxplot()+
  geom_jitter()+
  facet_grid(treatment~date)

# Part 2:  Population Information / Data Distribution
## Create a  box-plot across harvest dates showing distribution for each tray.   

## BOXPLOTS FOR LEAF FRESH WEIGHT COMPARISON
biomass_data %>% 
  mutate(treatment = fct_relevel(treatment,'80 ppm', '160 ppm', '280 ppm')) %>%
  ggplot(aes(treatment,LFW_g,color = treatment)) +
  geom_boxplot()+
  geom_jitter(width=0.15, alpha=0.5)+
  facet_wrap(~date, scales = 'free') +
  scale_color_brewer(palette = "Set1") + 
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 7,color="black"),
        axis.title.x = element_text(size=12,face="bold"),
        axis.title = element_text(size = 12, face="bold"),
        legend.position = "right",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="black", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(title="Harvest: LFW (g) by date ",
       fill="Tray",
       caption=NULL,
       x=NULL,
       y="LFW (g)")

## BOXPLOTS FOR LEAF AREA COMPARISON
biomass_data %>% 
  mutate(treatment = fct_relevel(treatment,'80 ppm', '160 ppm', '280 ppm')) %>%
  ggplot(aes(treatment,LA_mm2,color = treatment)) +
  geom_boxplot()+
  geom_jitter(width=0.15, alpha=0.5)+
  facet_wrap(~date, scales = 'free') +
  scale_color_brewer(palette = "Set1") + 
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 7,color="black"),
        axis.title.x = element_text(size=12,face="bold"),
        axis.title = element_text(size = 12, face="bold"),
        legend.position = "right",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="black", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(title="Harvest: Leaf Area (mm2) by date ",
       fill="Tray",
       caption=NULL,
       x=NULL,
       y="LA (mm2)")




# Timeline for mean values 
mean_values <- biomass_data %>% group_by(date,tray_id,treatment) %>%
  summarise(avg_LFW_g = mean(LFW_g),
            std_LFW_g = sd(LFW_g))

mean_values %>% mutate(treatment = fct_relevel(treatment,'80 ppm', '160 ppm', '280 ppm')) %>% 
  ggplot(aes(treatment,avg_LFW_g, fill = tray_id)) +
  geom_col()+
  geom_errorbar(aes(ymin = avg_LFW_g - std_LFW_g,
                    ymax = avg_LFW_g + std_LFW_g),width=0.3)+
  facet_wrap(~date, scales = 'free') +
  scale_fill_brewer(palette = "Set1")


# Boxplot for each  response variables: manual measurements (labels)
biomass_reshape <- melt(biomass_data, 
                    measure.vars = c("LFW_g", "LDW_g", "LA_mm2"))


biomass_reshape %>% 
  #mutate(treatment = fct_relevel(treatment,'80 ppm', '160 ppm', '280 ppm')) %>%
  mutate(harvest= as.factor(with(biomass_reshape, ifelse(date == '2023-02-02', 'H1',
                                               ifelse(date == '2023-02-09', 'H2', 'H3'))))) %>% 
  ggplot(aes(harvest,value,color = harvest)) +
  #geom_boxplot()+
  geom_jitter(width=0.15, alpha=0.5)+
  facet_wrap(variable~., scales = 'free') +
  scale_color_brewer(palette = "Set1") + 
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 7,color="black"),
        #axis.title.x = element_text(size=12,face="bold"),
        axis.title = element_text(size = 12, face="bold"),
        legend.position = "top",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="black", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(title="Biomass Acummulation Variables ",
       subtitle = "Manual measurements on each harvest day",
       color="Harvest Day",
       caption=NULL,
       x=NULL,
       y="Response Variable(g/g/mm2)")


# Leaf Fresh Weight datapoints per harvest day.
biomass_data %>% 
  #mutate(treatment = fct_relevel(treatment,'80 ppm', '160 ppm', '280 ppm')) %>%
  mutate(harvest= as.factor(with(biomass_data, ifelse(date == '2023-02-02', 'H1',
                                                         ifelse(date == '2023-02-09', 'H2', 'H3'))))) %>% 
  ggplot(aes(harvest,LFW_g,color = harvest)) +
  #geom_boxplot()+
  geom_jitter(width=0.15, alpha=0.5)+
  #facet_wrap(variable~., scales = 'free') +
  scale_color_brewer(palette = "Set1") + 
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 7,color="black"),
        #axis.title.x = element_text(size=12,face="bold"),
        axis.title = element_text(size = 12, face="bold"),
        legend.position = "top",
        legend.background = element_rect(fill="white",
                                         size=1, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="black", 
                                        size=1.5, linetype="solid"),
        strip.text.x = element_text(
          size = 12, color = "white", face = "bold.italic"))+
  labs(title="Biomass Acummulation Variables ",
       subtitle = "Manual measurements on each harvest day",
       color="Harvest Day",
       caption=NULL,
       x=NULL,
       y="Leaf Fresh Weight (g)")


