from shiny import App, render, ui,reactive
import pandas as pd
from pathlib import Path
import  matplotlib.pyplot as plt 
import shinyswatch
from shinywidgets import output_widget, render_widget
import plotly.express as px
import numpy as np
import matplotlib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# importing dataset-------------------------------------

all_incidents = pd.read_csv(Path(__file__).parent / "311_data.csv")
columns_to_convert = ['OPENEDDATETIME', 'SLA_Date', 'CLOSEDDATETIME']
all_incidents[columns_to_convert] = all_incidents[columns_to_convert].apply(pd.to_datetime,format="%Y-%m-%d")#origin input month/day/year,output year day month

# option lists---------------------------------------------------------------------------------------
options_category = all_incidents["Category"].sort_values().unique().tolist()
options_District = all_incidents["Council_District"].sort_values().unique().tolist()
options_casestatus = all_incidents["CaseStatus"].sort_values().unique().tolist()

#----color function----------------------------


#---------- ui input and output-----------------------------------------------------    

app_ui = ui.page_fluid(
             ui.layout_sidebar(  
                    ui.sidebar(
                    
                   
                    ui.input_selectize("category", ui.tags.h3("CATEGORY"), choices=options_category,selected=options_category,multiple=True),
                    #ui.input_selectize("district", ui.tags.h3("Council District"), choices=options_District,selected=options_District,multiple=True),
                    ui.input_selectize("casestatus", ui.tags.h3("Case Status"), choices=options_casestatus,selected=options_casestatus,multiple=True),

                    ui.input_date_range(
                        id="date_range",
                        label=ui.tags.h3("Date Range"),
                        start="2017-11-03", # Minimum date
                        end="2024-03-01", # Maximum date
                        min="2017-11-03", # Minimum date
                        max="2024-03-01", # Maximum date
                        format="yyyy-mm-dd",
                                separator=" to ",
                                ),              
                
                    ),
         #shinyswatch.get_theme('flatly'),
         
        ui.panel_title(ui.tags.h1("Summary of 311 Service in San Antonio",align='center'), window_title="311SanAntonio" ),
           
        ui.tags.h4("This report offers a thorough examination of 311 service requests received by the City of San Antonio. Utilizing data analysis,\
                   I will explore patterns related to request volume, response times, service categories, and departmental participation.\
                   The objective of this report is to provide a summary of the 311 Customer Service  requests \
                   and insights for enhancing service efficiency\
                                                                 "),
        
         
       
                           

                    # ----outputs---------
                     
                ui.tags.p(ui.output_ui("value_boxes"),),     
                ui.layout_columns(  
                           

                    ui.card(
                        ui.card_header(ui.tags.h2(" Service Volume and Trends ")),
                        output_widget("year_bar",width="auto", height="auto"),
                        output_widget("plot_daily_cases",width="auto", height="auto"),
                        output_widget("montly_bar",width="auto", height="auto"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header(ui.tags.h2("Response Times")), 
                        ui.output_data_frame("time_response_df"),
                        output_widget("response_category",width="auto", height="auto"),                      
                                              
                        output_widget("hist_response",width="auto", height="auto"), 
                        output_widget("source_type",width="auto", height="auto"),  
                        
                        
                        full_screen=True,               
                    ) ,   
                     ui.card(
                        ui.card_header(ui.tags.h2("Service Categories")),   
                        output_widget("plot_categories",width="auto", height="auto"),
                         output_widget("plot_dept",width="auto", height="auto"), 
                        full_screen=True,               
                    ) ,
                     
                    ui.card(
                        ui.card_header(ui.tags.h2("District Analysis")),   
                       
                        output_widget("plot_dist",width="auto", height="auto"),
                        output_widget("plot_dis_cat",width="auto", height="auto"),
                        full_screen=True,               
                    ) , 
                    
                    col_widths=(6,6,6,6) ,row_heights='grid-auto-rows' ,  
                 ),
                    
            ),
        )


def server(input, output, session):
    
    @reactive.calc()
    def filter_dataset():               
        ind_cat =all_incidents['Category'].isin(input.category())
        #ind_dist =all_incidents["Council_District"].isin(input.district())
        ind_case =all_incidents['CaseStatus'].isin(input.casestatus())
        ##
        selected_dates = input.date_range()  # Get the selected date range
        if selected_dates:  # Check if dates are selected
            start_date, end_date = selected_dates  # Unpack start and end dates
            ind_date = all_incidents['OPENEDDATETIME'].between(datetime.strptime(str(start_date),"%Y-%m-%d"),datetime.strptime(str(end_date),"%Y-%m-%d"))  # Filter data
        
        df_cat=all_incidents.loc[((ind_cat & ind_date) & ind_case),:]
        return df_cat
 
    #---------------------------------
    @reactive.calc
    def groups():
        return filter_dataset().groupby(filter_dataset()['OPENEDDATETIME'].dt.year)['CASEID'].count().reset_index().rename(columns={'OPENEDDATETIME':'Year','CASEID':'Count'})    
    @reactive.calc
    def groups_month():
        return filter_dataset().groupby(filter_dataset()['OPENEDDATETIME'].dt.month)['CASEID'].count().reset_index().rename(columns={'OPENEDDATETIME':'Month','CASEID':'Count'})    
    @reactive.calc
    def groups_category():
        return filter_dataset().groupby('Category')['CASEID'].count().reset_index().rename(columns={'CASEID':'Count'}).sort_values(by='Count')
    @reactive.calc
    def groups_dept():
        return filter_dataset().groupby('Dept')['CASEID'].count().reset_index().rename(columns={'Dept':'Department','CASEID':'Count'}).sort_values(by='Count')
    @reactive.calc
    def groups_dist():
        return filter_dataset().groupby('Council_District')['CASEID'].count().reset_index().rename(columns={'Council_District':'District','CASEID':'Count'}).sort_values(by='Count')

    @reactive.calc
    def groups_source():
        return filter_dataset().groupby('SourceID')['CASEID'].count().reset_index().rename(columns={'SourceID':'Source','CASEID':'Count'}).sort_values(by='Count')

    @reactive.calc
    def timeseries():
        df_time=filter_dataset().set_index('OPENEDDATETIME')
        #bd = pd.tseries.offsets.CustomBusinessDay(n=1,weekmask="Mon Tue Wed Thu Fri")
        #pd.tseries.offsets.BusinessDay(2)
        # Resample the DataFrame by day and apply mean aggregation
        daily_count = df_time.resample(pd.tseries.offsets.BusinessDay()).count().loc['2023-01-01':,'CASEID'].reset_index().rename(columns={'OPENEDDATETIME':'Request Date','CASEID':'Count'})
        return  daily_count
    @reactive.calc
    def response():
        df=filter_dataset() 
        df1=df.loc[df["CaseStatus"]=="Closed",'time_interval']
        
        return df1+1      
       
    ##############################################################################
    @output
    @render.ui
    def value_boxes():
        models = timeseries().agg(Total_Requests=("Count",'sum'),Daily_Requests =("Count",'mean'))
        ddd=[y for x, y in models.to_dict().items()][0]
        scores_by_model={x: int(y) for x, y in ddd.items()}
 

        return ui.layout_column_wrap(
            "  ",
            *[
                # For each model, return a value_box with the score, colored based on
                # how high the score is.
                ui.value_box(
                    ui.h4(model),
                    ui.h3(score),
                    theme="bg-gradient-orange-red",
                    full_screen=False,
                )
                for model, score in scores_by_model.items()
            ],"  ",
            #fixed_width=True,
            width="150px",
        )

    
    #-----------------------------------
    @output
    @render_widget       
    def year_bar(): 
        data=groups()
        data['Percentage'] =data.apply(lambda x:100*x/float(x.sum())).values[:,1]
        ax = px.bar(data,x='Year',y='Count',text=data['Percentage'].apply(lambda x: '{0:1.3f}%'.format(x)),
            title="Distribution of Requests By Year",)
        ax.update_layout(            
       font=dict( size=16),)
        #ax.show()         
        return ax   
    @output
    @render_widget
    def plot_daily_cases():
        df=timeseries()
        df['monthly_avg']=df['Count'].rolling(window=22).mean()
        pp=px.line(df, x='Request Date',y=['Count','monthly_avg'],         
           # height=300,width=400,template="none",
            orientation='h',   
            title="The Amount of Requests over Time",
            )
       
        pp.update_layout( font=dict(
        #family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"
        ),  margin=dict(l=30, r=20, t=70, b=10) ,
       #grid_xaxes=dict(dtick="M1",
       #     tickformat="%b\n%Y",
       #     ticklabelmode="period") ,
        ) 
        pp.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.00,
    xanchor="right",
    x=1,))    

        return pp      
    @output
    @render_widget
    def montly_bar():
        df=groups_month()
        df['Percentage'] =df.apply(lambda x:100*x/float(x.sum())).values[:,1]
               
        ax=px.line(df,x='Month',y='Count',
                   text=df['Percentage'].apply(lambda x: '{0:1.1f}%'.format(x)),
                   title="Distribution of Requests by Month")  
        ax.update_layout(            
        font=dict( size=16), xaxis=dict(
        tickvals=df['Month'].values,  # Set tick positions for all categories
        ticktext=df['Month'].values,  # Set tick labels for all categories
        #tickangle=-45  # Optional: Rotate long labels for better readability
        ),)
    
        return ax
    #--------------------------------------------
    @output
    @render_widget
    def source_type():
        
        pi=px.pie(groups_source(),values='Count',names='Source', 
                                    title="Distribution of the Service Sources",
                                                                )
        pi.update_layout(margin=dict(t=70, b=0, l=0, r=0))
        pi.update_layout( font=dict(size=16))
        return pi        
        
        
        
    @render.data_frame
    def time_response_df():        
        dd=response().describe().map(lambda x: f"{x:0.2f}").reset_index().T
        dd.columns = dd.iloc[0]        
        return dd.drop(dd.index[0])
    @output
    @render_widget   
    def hist_response():
        
        df=filter_dataset().groupby('CaseStatus')['CASEID'].count().reset_index().rename(columns={'CaseStatus':'Status','CASEID':'Count'}).sort_values(by='Count')

        pi=px.pie(df,values='Count',names='Status', 
                                    title="Distribution of Requests By Status",)
                                                                
        pi.update_layout(margin=dict(t=70, b=0, l=0, r=0))
        pi.update_layout( font=dict(size=16))          
        pi.update_layout(font=dict( size=16))        
        return pi
    
    
    @output
    @render_widget   
    def response_category(): 
        
        quantile_counts = filter_dataset().groupby('time_interval_quantiles')['CASEID'].count().reset_index().rename(columns={'CASEID':'Count'})

        pii=px.pie(quantile_counts,values='Count',names='time_interval_quantiles', 
                               title="Distribution of Time Response",
                               #width='100%',
                               #height='100%',
                               #category_orders=['1 Day', '1-3 Days', '4-15Days' , '16-30 Days', '31-364' ,  '365-1000' , '1001-1959 Days'],
                               )
        pii.update_layout(margin=dict(t=50, b=0, l=0, r=0))
        pii.update_layout(font=dict( size=16))
        
        return pii
       

    
    #----------------------------------------------
    @output
    @render_widget
    def plot_categories():       
        data=groups_category()
        data['Percentage'] = data[['Count']].apply(lambda x:100*x/float(x.sum())).values
        ax = px.bar(data,y='Category',x='Count',text=data['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
            title="Distribution of Requests By Category",orientation='h',
            color_discrete_sequence=px.colors.qualitative.Dark2_r)
        ax.update_layout(            
       font=dict( size=16),)
        return ax
        
    
    @output
    @render_widget
    def plot_dept():       
        data=groups_dept()
        data['Percentage'] = data[['Count']].apply(lambda x:100*x/float(x.sum())).values
        ax = px.bar(data,y='Department',x='Count',text=data['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
            title="Distribution of Requests By Department",orientation='h',
            color_discrete_sequence=px.colors.qualitative.Dark2_r)
        ax.update_layout(font=dict( size=16),)
        return ax
    #####-----------------------------------------------------
    @output
    @render_widget
    def plot_dist():       
        data=groups_dist()
        data['Percentage'] = data[['Count']].apply(lambda x:100*x/float(x.sum())).values
        ax = px.bar(data,x='District',y='Count',text=data['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
            title="Distribution of Requests By District",orientation='v',
            color_discrete_sequence=px.colors.qualitative.Set2)
        
        ax.update_layout(            
       font=dict( size=16), xaxis=dict(
        tickvals=data['District'].values,  # Set tick positions for all categories
        ticktext=data['District'].values,  # Set tick labels for all categories
        #tickangle=-45  # Optional: Rotate long labels for better readability
        ),)
        return ax
    @output
    @render_widget
    def plot_dis_cat(): 
        CD_CA=filter_dataset().groupby(['Council_District','Category'])['CASEID'].count().unstack()
        ax=px.imshow(CD_CA.fillna(0)/1000,text_auto='0.3f',
                     aspect="auto",
                     title="Distribution of Categories by Council District",)
        ax.update_yaxes(tickvals=CD_CA.index,ticktext=CD_CA.index)
        ax.update_layout(coloraxis_colorbar_title="Count/1000")
        ax.update_layout( font=dict(size=16))
        ax.update_layout(xaxis_tickangle=45)
        
        
        return ax
    
    
    
    #---------------------------------

    
    
          
# to add a static doc such as logo or img
www_dir = Path(__file__).parent / "www"
app = App(app_ui, server,static_assets=www_dir)

#rsconnect deploy shiny C:\Users\Rayan\OneDrive\GreatLearning_DSBA\business_analytics\SanAntonioCity311\311_SA_APP  --name keep-learning --title 311_SA_app