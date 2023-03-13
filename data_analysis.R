library(tidyverse)
library(shiny)

plant_data <- read.csv("C:\\Users\\saaha\\OneDrive\\Documents\\GitHub\\BiomassPredictonAI\\harvest_bydate.csv")

View(plant_data)

names <- setdiff(names(plant_data), "Species")

#ggplot(data = plant_data) + geom_point(mapping = aes(x = plant_area, y = LDW_g))

#runExample("01_hello")

ui <- fluidPage(
  
  titlePanel("Plant Data Correlation Analysis"),
  
    
    sidebarPanel(
      selectInput('xcol', 'X Variable', vars),
      selectInput("ycol", 'Y Variable', vars)
    ),
    
  mainPanel(
    plotOutput('dataplot')
  )
  
)

server <- function(input, output) {
  
}

#shinyApp(ui = ui, server = server)
