#!/usr/bin/env Rscript

load_packages <- function(required_packages) {
  for (p in required_packages) {
    if (!require(p, character.only = TRUE)) install.packages(p, repos="https://cloud.r-project.org")
    library(p, character.only = TRUE)
  }
}

load_packages(c("shiny", "ggplot2", "reshape2"))

update_script_parameters <- function() {
  #browser()
  args <- commandArgs(trailingOnly=TRUE)
  if (length(args)==0) {
    #stop("At least one argument must be supplied (path to file with logs).", call.=FALSE)
    args <- "../TEMP/score_ref.csv"
  }
  score_ref_filename <<- args[1]
  if (length(args) > 1) {
    shiny_port <<- strtoi(args[2])
  }
  else {
    shiny_port <<- 6006
  }
}

update_log_list <- function(score_ref_filename) {
  #browser()
  score_ref_data <<- read.delim(score_ref_filename,
                               header = FALSE,
                               col.names = c("name", "score_log_path", "hyper_log_path"),
                               stringsAsFactors = FALSE)
  score_ref_dirname <- dirname(score_ref_filename)
  score_ref_data$score_log_path <<- sapply(
    score_ref_data$score_log_path,
    function(x) ifelse(is.na(x) || file.exists(x), x, file.path(score_ref_dirname, x)))
  score_ref_data$hyper_log_path <<- sapply(
    score_ref_data$hyper_log_path,
    function(x) ifelse(is.na(x) || file.exists(x), x, file.path(score_ref_dirname, x)))
  log_task_names <<- 1:length(score_ref_data$name)
  names(log_task_names) <<- score_ref_data$name
  selected_log_task_number <<- ifelse(length(score_ref_data$name) > 0, 1L, 0L)
  log_list_just_loaded <<- TRUE
}

update_log_data <- function() {
  #browser()
  update_log_data_clear <- function() {
    score_log_data_valid <<- FALSE
    log_variable_names <<- character()
    log_attempt_numbers <<- integer()
    selected_log_attempt_numbers <<- integer()
  }
  if (selected_log_task_number == 0L) {
    update_log_data_clear()
    return()
  }
  score_log_filename <- score_ref_data$score_log_path[selected_log_task_number]
  score_log_data_valid <<- file.exists(score_log_filename)
  if (!score_log_data_valid) {
    update_log_data_clear()
    return()
  }
  tryCatch({
    score_log_data <<- read.delim(score_log_filename)
    score_log_data <<- score_log_data[!duplicated(score_log_data[,c("Attempt","Epoch")], fromLast = TRUE),]
  }, error = function(e) {
    update_log_data_clear()
    return()
  })
  log_variable_names <<- setdiff(colnames(score_log_data), c("Attempt","Epoch"))
  log_attempt_numbers <<- unique(score_log_data[,"Attempt"])
  #selected_log_attempt_numbers <<- ifelse(length(log_attempt_numbers) > 0, log_attempt_numbers[1], c())
  
  update_hyper_log_data <- function() {
    hyper_log_filename <- score_ref_data$hyper_log_path[selected_log_task_number]
    if (is.na(hyper_log_filename)) {
      hyper_log_data <<- data.frame()
      return(FALSE)
    }
    hyper_log_data_valid <- file.exists(hyper_log_filename)
    if (!hyper_log_data_valid) {
      hyper_log_data <<- data.frame()
      return(FALSE)
    }
    tryCatch({
      hyper_log_data <<- read.delim(hyper_log_filename)
    }, error = function(e) {
      hyper_log_data <<- data.frame()
      return(FALSE)
    })
    TRUE
  }
  
  if (update_hyper_log_data()) {
    hyper_log_data <<- hyper_log_data[order(-hyper_log_data$target),] 
    iters <- unique(hyper_log_data$iter)
    #browser()
    if (length(log_attempt_numbers) > 0) {
      log_attempt_number_last <- log_attempt_numbers[length(log_attempt_numbers)]
      log_attempt_numbers <<- intersect(iters, log_attempt_numbers)
      if (!(log_attempt_number_last %in% log_attempt_numbers)) {
        log_attempt_numbers <<- c(log_attempt_number_last, log_attempt_numbers)
      }
    }
  }
  
  if (log_list_just_loaded) {
    log_list_just_loaded <<- FALSE
    variables_choices <<- log_variable_names
    attempts_choices <<- log_attempt_numbers
  }
}

update_script_parameters()
update_log_list(score_ref_filename)
update_log_data()

ui <- fluidPage(
   titlePanel("Score log observer"),
   
   sidebarLayout(
      sidebarPanel(
        selectInput(inputId = "task_name",
                    label = "Task name:",
                    choices = log_task_names),
        checkboxGroupInput(inputId = "plot_opts",
                           label = "Plot options:",
                           choices = c("show raw data" = 1),
                           selected = character()),
        checkboxGroupInput(inputId = "variables",
                           label = "Variables:",
                           choices = variables_choices,
                           selected = variables_choices),
        selectInput(inputId = "attempts",
                    label = "Attempts:",
                    choices = log_attempt_numbers),
        actionButton(inputId = "refresh_btn",
                     label = "Refresh"),
        width = 3
      ),
      
      # Show a plot of the generated distribution
      mainPanel(
        plotOutput("plot1", click = "plot_click"),
        verbatimTextOutput("info"),
        tableOutput("hyper_log_table"),
        width = 9
      )
   )
)

server <- function(input, output, session) {

  update_sized_panel <- function() {
    #browser()
    update_log_data()
    if (!all(log_variable_names == variables_choices)) {
      variables_choices <<- log_variable_names
      updateCheckboxInput(session,
                          inputId = "variables",
                          value = variable_choices)
    }
    if (!all(log_attempt_numbers == attempts_choices)) {
      attempts_choices <<- log_attempt_numbers
      updateSelectInput(session,
                        inputId = "attempts",
                        choices = attempts_choices)
    }
  }
  
  observe({
    #browser()
    if (input$task_name != selected_log_task_number)
    {
      selected_log_task_number <<- strtoi(input$task_name)
      update_sized_panel()
    }
  })

  observe({
    #browser()
    input$refresh_btn
    update_sized_panel()
  })
  
  output$plot1 <- renderPlot({
    #browser()
    input$refresh_btn
    if (length(input$variables) == 0 || length(input$attempts) == 0) {
      p <- ggplot()
      melted <<- data.frame(Epoch=numeric(), value=numeric())
    }
    else {
      log_data_subset <<- score_log_data[score_log_data["Attempt"] == strtoi(input$attempts), c("Epoch", input$variables)]
      melted <<- melt(log_data_subset, id.vars="Epoch")
      p <- ggplot(data=melted, aes(x=Epoch, y=value, colour=variable)) + geom_smooth()
      if ("1" %in% input$plot_opts) {
        p <- p + geom_point() + geom_line()
      }
    }
    p
  })
  
  output$info <- renderPrint({
    nearPoints(melted, input$plot_click)
  })

  output$hyper_log_table <- renderTable(
    {
      #browser()
      input$task_name
      input$refresh_btn
      hyper_log_data
    },
    striped = TRUE,
    hover = TRUE,
    bordered = TRUE)
}

app <- shinyApp(ui = ui, server = server)
runApp(app, port = shiny_port)