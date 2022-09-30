

from botbuilder.core import ActivityHandler, TurnContext, MessageFactory, ConversationState, UserState, Storage
from botbuilder.schema import ChannelAccount
from data_models import ConversationFlow, Question, UserInputs
from config import DefaultConfig
from aiohttp import web
from botbuilder.core.integration import aiohttp_error_middleware

CONFIG = DefaultConfig()

import nltk
nltk.download('all')

import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from keras.models import load_model
import regex as re
import asyncio
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot
from sorted_months_weekdays import *
from sort_dataframeby_monthorweek import *

model = load_model('trained_chatbot_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

data_Finance =pd.read_excel("Financial Information (1).xlsx", sheet_name="Financial Information")
data_Budget =pd.read_excel("Financial Information (1).xlsx", sheet_name="Budget Information")
data_Hotel_1 =pd.read_excel("Financial Information (1).xlsx", sheet_name="Hotel Information-1")
data_Hotel =pd.read_excel("Financial Information (1).xlsx", sheet_name="Hotel Information-2")

columns_list1 = ['CY','LY']
for i in columns_list1:
     data_Finance[i] = np.where(data_Finance['Category']=='Total Revenue',data_Finance[i]*(-1),data_Finance[i])

columns_list2 = ['CY Budget','Last Year Budget']
for i in columns_list2:
     data_Budget[i] = np.where(data_Budget['Category']=='Total Revenue',data_Budget[i]*(-1),data_Budget[i])     

data_Hotel_fin = pd.merge(data_Hotel_1,data_Hotel[['Hotel','Total Rooms']],on = 'Hotel')

class ValidationResult:
    def __init__(
        self, is_valid: bool = False, value: object = None, message: str = None
    ):
        self.is_valid = is_valid
        self.value = value
        self.message = message




class MyBot(ActivityHandler):
   
    
    def __init__(self, conversation_state: ConversationState, user_state: UserState):
        if conversation_state is None:
            raise TypeError(
                "[CustomPromptBot]: Missing parameter. conversation_state is required but None was given"
            )
        if user_state is None:
            raise TypeError(
                "[CustomPromptBot]: Missing parameter. user_state is required but None was given"
            )

        self.conversation_state = conversation_state
        self.user_state = user_state

        self.flow_accessor = self.conversation_state.create_property("ConversationFlow")
        self.profile_accessor = self.user_state.create_property("UserInputs")
    
    # Functions defined to predict the response class of the user query
    
    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words


    def bow(self, sentence, words, show_details=True):
        # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))


    def predict_class(self, sentence, model):
        # filter out predictions below a threshold
        p = self.bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
        
     
    # Functions defined to validate the user inputs
     
    def _validate_name(self, user_input: str) -> ValidationResult:
        if not user_input:
            return ValidationResult(
                is_valid=False,
                message="Please enter a name that contains at least one character.",
            )

        return ValidationResult(is_valid=True, value=user_input)

    def _validate_hotel(self, user_input: str) -> ValidationResult:
       hotel_list = data_Hotel_fin.Hotel.unique().tolist() 
       if user_input not in hotel_list:
            return ValidationResult(
                is_valid=False,
                message="The available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again.",
            )

       return ValidationResult(is_valid=True, value=user_input)

    def _validate_date(self, user_input: str) -> ValidationResult:
        if not user_input:
            return ValidationResult(
                is_valid=False,
                message="Please enter a month that contains at least one character.",
            )

        return ValidationResult(is_valid=True, value=user_input)     # Month is case sensitive

    # KPI functions for bot responses
    # Any new KPI function or any bot response patterns can be defined here
    # For every new bot response pattern, you must add an 'intent' to the intents.json file and re-train the model using the 'chatbot_model.py' file and register new 'classes.pkl' and 'words.pkl'

    @asyncio.coroutine
    async def goodbye_func(self, hotel,month, turn_context:TurnContext):
       output = str("Happy to help! :)")
       await turn_context.send_activity(MessageFactory.text(output))
       await turn_context.send_activity(MessageFactory.text("Please type 'BATMAN' if you wish to restart the session.."))
    
    @asyncio.coroutine
    async def unknown_func(self, hotel,month, turn_context:TurnContext):
       await turn_context.send_activity(MessageFactory.text("Ummm... I am not sure I understand. Can you please say that again?"))

    
    @asyncio.coroutine
    async def restart_func(self, hotel,month, turn_context:TurnContext):
        await self.conversation_state.clear_state(turn_context)
        await self.user_state.clear_state(turn_context)
        await turn_context.send_activity(MessageFactory.text("Type anything to continue..."))
                 

    @asyncio.coroutine
    async def bussi_prom_expenses_func(self, hotel,month, turn_context:TurnContext):
        exp_cur = data_Finance.loc[(data_Finance["Hotel"]== hotel) & (data_Finance["Sub Category"]== "Business Promotion Expenses"), 'CY'].sum()
        exp_bud = data_Budget.loc[(data_Budget["Hotel"]== hotel) & (data_Budget["Sub Category"]== "Business Promotion Expenses"), 'CY Budget'].sum()
        op1 = round(exp_cur)
        op2 = round(exp_bud)
        diff = int(op2-op1)
        if diff > 0:
            op3 = str("The Business Promotion Expenses for {} is ${:,}, which is ${:,} under-budget for the current year").format(hotel,op1,diff)
        else:
            diff = -1*diff
            op3 = str("The Business Promotion Expenses for {} is ${:,}, which is ${:,} over-budget for the current year").format(hotel,op1,diff)
        
        output = str("The actual budget for {} was ${:,}".format(hotel,op2))
        await turn_context.send_activity(MessageFactory.text(op3))
        await turn_context.send_activity(MessageFactory.text(output))
        
        
    
    @asyncio.coroutine
    async def fnb_revenue_func(self, hotel,month, turn_context:TurnContext):
        new_df = data_Finance[data_Finance["Sub Category"]== "F&B Revenue"]
        total_rev = new_df.loc[(new_df["Hotel"]== hotel) & (new_df["Month"]== month)& (new_df["Sub Category"]== "F&B Revenue"), 'CY'].sum()
        import plotly.express as px                                         
        rev_list = new_df['Gl Name'].unique().tolist()        
        fig = px.pie(new_df, values='CY', names='Gl Name', title='Food & Beverages Revenue Distribution')
        fig.update_traces(textposition='inside')
        fig.update_layout(height=400,width=430,uniformtext_minsize=10, uniformtext_mode='hide',legend=dict(font=dict(size=12)),margin=dict(l=0,r=0,b=0,t=50,pad=0))
        img = plot(fig,filename='fnb.html',config={'displayModeBar':True}, output_type = 'div')
        output = str("The Food and Beverage related revenue for {} in the month of {} is ${:,}.").format(hotel,month,round(total_rev))
        await turn_context.send_activity(MessageFactory.text(output))    
        await turn_context.send_activity(MessageFactory.text("A detailed bifurcation is available in your browser for viewing."))
        displayHTML(img)
        
                
        
    @asyncio.coroutine
    async def rooms_sold_func(self, hotel,month, turn_context:TurnContext):
        rooms = data_Hotel_fin.loc[(data_Hotel_fin["Month"]== month) & (data_Hotel_fin["Hotel"]== hotel), 'Room Sold'].sum()
        output = str("The total Rooms sold at {} in the month of {} is {}".format(hotel, month, rooms))
        await turn_context.send_activity(MessageFactory.text(output))
    
    
    @asyncio.coroutine        
    async def occupancy_rate_func(self, hotel,month, turn_context:TurnContext): #Occupancy Rate =  Total number of occupied rooms (Rooms sold) / Total number of available rooms

        hotel_list = data_Hotel_fin.Hotel.unique().tolist()
        month_list = data_Hotel_fin.Month.unique().tolist()

        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')

        if hotel.lower().replace(' ','') not in hotel_list:
            print('\nThe available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            occupancy_rate()
        else:
            data_Hotel_fin_v1 = data_Hotel_fin[(data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')) & (data_Hotel_fin.Month==month)]
            num = data_Hotel_fin_v1['Room Sold']
            den = data_Hotel_fin_v1['Total Rooms']
            occ_rate = round(num/den,4)*100
            occ_rate = occ_rate.to_string(index=False)
            output1 = str("The Occupancy Rate for {} in the month of {} for the current year is {}%".format(hotel,month, occ_rate))


            rate = pd.DataFrame((data_Hotel_fin[data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')].groupby('Month')['Room Sold'].sum()/data_Hotel_fin[data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')].groupby('Month')['Total Rooms'].sum())*100)
            rate.rename({0:'Occupancy_Rate'},axis=1,inplace=True)
            rate['Occupancy_Rate'] = round( rate['Occupancy_Rate'],2)

            output2 = str(' The overall occupancy rate for this property in the current year is '+str(round(rate['Occupancy_Rate'].mean(),0))+'%')


            # Overall insights

            # Best hotel and comparison with your hotel

            data_Hotel_fin_w1=data_Hotel_fin.groupby('Hotel').agg({'Room Sold':'sum','Total Rooms':'sum'}).reset_index()
            data_Hotel_fin_w1['Occupancy_Rate']=round((data_Hotel_fin_w1['Room Sold']/data_Hotel_fin_w1['Total Rooms'])*100,2)
            data_Hotel_fin_w1=data_Hotel_fin_w1[['Hotel','Occupancy_Rate']]
            data_Hotel_fin_w2=data_Hotel_fin_w1[data_Hotel_fin_w1.Occupancy_Rate == data_Hotel_fin_w1.Occupancy_Rate.max()]
            data_Hotel_fin_w3=data_Hotel_fin_w1[data_Hotel_fin_w1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=data_Hotel_fin_w2.Hotel.sum()
            j=data_Hotel_fin_w2.Occupancy_Rate.sum()
            i=str(j)+'%'
            u=data_Hotel_fin_w3.Hotel.sum()
            m=data_Hotel_fin_w3.Occupancy_Rate.sum()
            k=round(j-m,2)
            k=str(k)+'%'

            if q==u:
                output9=str("1. The property {} you are interested in is the best performing hotel, with an occupancy rate of {}".format(q,i))
            else:
                output9=str("1. The best performing property is: {}. {} has {} less occupancy rate compared to the best performing hotel".format(q,hotel,k))


            # Hotel wise Insights

            # Best performing quarter

            data_Hotel_fin_v1 = data_Hotel_fin[(data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            cond = [data_Hotel_fin_v1.Month =='Jan',data_Hotel_fin_v1.Month =='Feb',data_Hotel_fin_v1.Month =='Mar',data_Hotel_fin_v1.Month =='Apr',data_Hotel_fin_v1.Month =='May',data_Hotel_fin_v1.Month =='Jun',data_Hotel_fin_v1.Month =='Jul',data_Hotel_fin_v1.Month =='Aug',data_Hotel_fin_v1.Month =='Sep',data_Hotel_fin_v1.Month =='Oct',data_Hotel_fin_v1.Month =='Nov',data_Hotel_fin_v1.Month =='Dec']
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Hotel_fin_v1['Quarter']=np.select(cond,val)

            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Quarter').agg({'Room Sold':'sum','Total Rooms':'sum'}).reset_index()
            data_Hotel_fin_v2['Occupancy_Rate']=data_Hotel_fin_v2['Room Sold']/data_Hotel_fin_v2['Total Rooms']
            data_Hotel_fin_v3=data_Hotel_fin_v2[data_Hotel_fin_v2.Occupancy_Rate==data_Hotel_fin_v2.Occupancy_Rate.max()]
            x=data_Hotel_fin_v3['Quarter'].sum()
            output3 = str("I have collated some interesting insights about the property {} -".format(hotel))
            output4 = str("2. The property had the best performance in {}".format(x))

            # Best Performing month

            data_Hotel_fin_v1 = data_Hotel_fin[(data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Month').agg({'Room Sold':'sum','Total Rooms':'sum'}).reset_index()
            data_Hotel_fin_v2['Occupancy_Rate']=data_Hotel_fin_v2['Room Sold']/data_Hotel_fin_v2['Total Rooms']
            data_Hotel_fin_v3=data_Hotel_fin_v2[data_Hotel_fin_v2.Occupancy_Rate==data_Hotel_fin_v2.Occupancy_Rate.max()]
            x=data_Hotel_fin_v3['Month'].sum()
            x_mean = data_Hotel_fin_v2['Occupancy_Rate'].mean()
            data_Hotel_fin_vx=data_Hotel_fin_v2[data_Hotel_fin_v2.Month==x]
            diff = round((data_Hotel_fin_vx.Occupancy_Rate.sum()-x_mean)*100,0)
            diff = str(diff) + ' %'
            output5 = str("3. The property had the best occupancy rate in month {}, making it the toughest month for a customer to get a room. The hotel had {} higher occupancy than average.".format(x,diff))

            # Worst Performing month

            data_Hotel_fin_v1 = data_Hotel_fin[(data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Month').agg({'Room Sold':'sum','Total Rooms':'sum'}).reset_index()
            data_Hotel_fin_v2['Occupancy_Rate']=data_Hotel_fin_v2['Room Sold']/data_Hotel_fin_v2['Total Rooms']
            data_Hotel_fin_v3=data_Hotel_fin_v2[data_Hotel_fin_v2.Occupancy_Rate==data_Hotel_fin_v2.Occupancy_Rate.min()]
            x=data_Hotel_fin_v3['Month'].sum()
            x_mean = data_Hotel_fin_v2['Occupancy_Rate'].mean()
            data_Hotel_fin_vx=data_Hotel_fin_v2[data_Hotel_fin_v2.Month==x]
            diff = round((x_mean-data_Hotel_fin_vx.Occupancy_Rate.sum())*100,0)
            diff = str(diff) + ' %'
            output6 = str("4. The property had the worst occupancy rate in month {}, making it the easiest month for a customer to get a room. The hotel had {} lower occupancy than average.".format(x,diff))

            # Month Over Month highest decrease and increase
            data_Hotel_fin_v1 = data_Hotel_fin[(data_Hotel_fin.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Month').agg({'Room Sold':'sum','Total Rooms':'sum'}).reset_index()
            data_Hotel_fin_v2['Occupancy_Rate']=round((data_Hotel_fin_v2['Room Sold']/data_Hotel_fin_v2['Total Rooms'])*100,0)
            data_Hotel_fin_v2=Sort_Dataframeby_Month(data_Hotel_fin_v2,monthcolumnname='Month')
            data_Hotel_fin_v2=data_Hotel_fin_v2.reset_index()
            data_Hotel_fin_v2['index_1'] = data_Hotel_fin_v2['index']+1
            data_Hotel_fin_v3 = pd.merge(data_Hotel_fin_v2[['Month','Occupancy_Rate','index']],data_Hotel_fin_v2[['Month','Occupancy_Rate','index_1']],left_on='index',right_on='index_1',how='left')
            data_Hotel_fin_v3.rename({'Month_x':'Month','Occupancy_Rate_x':'CM','Occupancy_Rate_y':'LM'},axis=1,inplace=True)
            data_Hotel_fin_v3=data_Hotel_fin_v3[['Month','CM','LM','index']]
            data_Hotel_fin_v3['LM']=np.where(data_Hotel_fin_v3.LM.isnull(),data_Hotel_fin_v3.CM,data_Hotel_fin_v3.LM)
            data_Hotel_fin_v3['MOM'] = data_Hotel_fin_v3['CM']-data_Hotel_fin_v3['LM']
            data_Hotel_fin_vx=data_Hotel_fin_v3[data_Hotel_fin_v3.MOM==data_Hotel_fin_v3.MOM.min()]
            data_Hotel_fin_vy=data_Hotel_fin_v3[data_Hotel_fin_v3.MOM==data_Hotel_fin_v3.MOM.max()]
            a=data_Hotel_fin_vx.Month.sum()
            b=round(abs(data_Hotel_fin_vx.MOM.sum()),0)
            b=str(b)+'%'
            c=data_Hotel_fin_vx['index'].sum()
            d=data_Hotel_fin_v3[data_Hotel_fin_v3['index']==(c-1)].Month.sum()

            e=data_Hotel_fin_vy.Month.sum()
            f=round(abs(data_Hotel_fin_vy.MOM.sum()),0)
            f=str(f)+'%'
            g=data_Hotel_fin_vy['index'].sum()
            h=data_Hotel_fin_v3[data_Hotel_fin_v3['index']==(g-1)].Month.sum()


            output7=str("5. The highest Month over Month decrease in occupancy rate occured from {} to {}, where the value fell by {}. The property needs to check up on why there is a sudden dip.".format(d,a,b))
            output8=str("6. The highest Month over Month increase in occupancy rate occured from {} to {}, where the value grew by {}. The property needs to prepare themselves to handle the surge of customers.".format(h,e,f))
            data = [go.Scatter(x=data_Hotel_fin_v2['Month'],y=data_Hotel_fin_v2['Occupancy_Rate'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            img = plot(fig,filename='occrate.html',config={'displayModeBar':True}, output_type = 'div')
            
            fin1 = output1+output2 
            fin2 = os.linesep+os.linesep+output3+os.linesep+os.linesep+os.linesep+os.linesep+output9+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8
            img_text = str("I have created a trend chart for the performance indicator - Occupancy Rate, it is open and available in your browser for viewing")
            
            
            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img)
            
    @asyncio.coroutine  
    async def ebitdaP_func(self,hotel,month,turn_context:TurnContext): # EBIDTA = GOP - Total Expense
        
        
        hotel_list = data_Hotel_fin.Hotel.unique().tolist()
        month_list = data_Hotel_fin.Month.unique().tolist()
        
        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')
            
        if hotel.lower().replace(' ','') not in hotel_list:
            print('\nThe available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            ebitdaP_func()
        else:
            # Current Year Current Month EBIDTA%
            
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_Finance_v1 = data_Finance[data_Finance['Month']==month]
            revenue_c = data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].CY.sum()
            expenses_c = data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].CY.sum()
            gop_c = revenue_c-expenses_c
            
            expense_c = data_Finance_v1[(data_Finance_v1['Category']=='Total Expense')].CY.sum()
            ebitda_c = round(gop_c-expense_c,0)
            ebitda_c_view=f"{ebitda_c:,}"
            ebitdap_c =round((ebitda_c/revenue_c),0)*100
            
            
            output1 = str("The EBIDTA for {} in the month of {} for the current year is {}".format(hotel,month, ebitda_c_view))
            output2 = str(" The EBIDTA% for {} in the month of {} for the current year is {} %".format(hotel,month, ebitdap_c))
            
            # INSIGHTS
            
            # Overall insights

            # Best hotel and comparison with your hotel

            data_Finance_v2 = data_Finance.copy()
            revenue=data_Finance_v2[data_Finance_v2['Category']=='Total Revenue'].groupby('Hotel').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v2[data_Finance_v2['Category']=='Total Expenses'].groupby('Hotel').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v2[data_Finance_v2['Category']=='Total Expense'].groupby('Hotel').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Hotel')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Hotel')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3['EBIDTA'] = round(data_Finance_v3['GOP'] - data_Finance_v3['Expense'],0)
            data_Finance_v3=data_Finance_v3[['Hotel','EBIDTA']]
            ebidta_max=data_Finance_v3[data_Finance_v3.EBIDTA == data_Finance_v3.EBIDTA.max()]
            ebidta_cur = data_Finance_v3[data_Finance_v3['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=ebidta_max.Hotel.sum()
            j=ebidta_max.EBIDTA.sum()
            i=f"{j:,}"
            u=ebidta_cur.Hotel.sum()
            m=ebidta_cur.EBIDTA.sum()
            k=round(j-m,0)
            k=f"{k:,}"
            m_view = f"{m:,}"
            
            output3 = str("I have collated some interesting insights about the property {}".format(hotel))

            if q==u:
                output4=str("1. The property {} you are interested in is the best performing hotel, with EBIDTA of {}".format(q,i))
            else:
                if m >= 0:
                    output4=str("1. The best performing property is {}. {} has {} less EBIDTA compared to the best performing hotel".format(q,hotel,k))
                else:
                    output4=str("1. The best performing property is {} with EBIDTA {}, while {} has EBIDTA {}, meaning a poor cash flow as EBIDTA is negative ".format(q,i,hotel,m_view))
            
            
            # Hotel wise insights
            
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            
            # EBIDTA in the LY same month
            
            data_Finance_vly = data_Finance[data_Finance['Month']==month]
            revenue_l = data_Finance_vly[data_Finance_vly['Category']=='Total Revenue'].LY.sum()
            expenses_l = data_Finance_vly[data_Finance_vly['Category']=='Total Expenses'].LY.sum()
            gop_l = revenue_l-expenses_l
            
            expense_l = data_Finance_vly[(data_Finance_vly['Category']=='Total Expense')].LY.sum()
            ebitda_l = round(gop_l-expense_l,0)
            ebitda_l_view=f"{ebitda_l:,}"
            
            if  ebitda_c>ebitda_l:
                output5=str("2. There is an increase in EBIDTA from last year to this year in {} in {} from {} to {}.".format(hotel,month,ebitda_l_view,ebitda_c_view))
            elif ebitda_c<ebitda_l:
                output5=str("2. There is a decrease in EBIDTA from last year to this year in {} in {} from {} to {}.".format(hotel,month,ebitda_l_view,ebitda_c_view))
            else:
                output5=str("2. There is no change in EBIDTA from last year to this year in {} in {} with EBIDTA being {}.".format(hotel,month,ebitda_c_view))

     
            # EBIDTA in the LY - Overall
            
            revenue_lo = data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].LY.sum()
            expenses_lo = data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].LY.sum()
            gop_lo = revenue_lo-expenses_lo
            
            expense_lo = data_Finance_v1[(data_Finance_v1['Category']=='Total Expense')].LY.sum()
            ebitda_lo = round(gop_lo-expense_lo,0)
            ebitda_lo_view=f"{ebitda_lo:,}"
            
            if  m>ebitda_lo:
                output6=str("3. There is an increase in EBIDTA from last year to this year in {} from {} to {}.".format(hotel,ebitda_lo_view,m_view))
            elif m<ebitda_lo:
                output6=str("3. There is a decrease in EBIDTA from last year to this year in {} from {} to {}.".format(hotel,ebitda_lo_view,m_view))
            else:
                output6=str("3. There is no change in EBIDTA from last year to this year in {} with EBIDTA being {}.".format(hotel,ebitda_lo_view))
            
            
            # Month with highest EBIDTA
            revenue=data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].groupby('Month').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].groupby('Month').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v1[data_Finance_v1['Category']=='Total Expense'].groupby('Month').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Month')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Month')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3['EBIDTA'] = round(data_Finance_v3['GOP'] - data_Finance_v3['Expense'],0)
            data_Finance_v3=data_Finance_v3[['Month','EBIDTA']]
            data_Finance_v4=data_Finance_v3[data_Finance_v3.EBIDTA==data_Finance_v3.EBIDTA.max()]
            y=data_Finance_v4.EBIDTA.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output7 = str("4. The property had the best EBIDTA of {} in month {}, making it the month with highest cash flow.".format(y,x))

            # Month with lowest EBIDTA
            revenue=data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].groupby('Month').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].groupby('Month').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v1[data_Finance_v1['Category']=='Total Expense'].groupby('Month').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Month')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Month')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3['EBIDTA'] = round(data_Finance_v3['GOP'] - data_Finance_v3['Expense'],0)
            data_Finance_v3=data_Finance_v3[['Month','EBIDTA']]
            data_Finance_v4=data_Finance_v3[data_Finance_v3.EBIDTA==data_Finance_v3.EBIDTA.min()]
            y=data_Finance_v4.EBIDTA.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output8 = str("4. The property had the lowest EBIDTA of {} in month {}, making it the month with lowest cash flow.".format(y,x))
            
            # Quarter with best performance
            cond = [data_Finance_v3.Month =='January',
            data_Finance_v3.Month =='February',
            data_Finance_v3.Month =='March',
            data_Finance_v3.Month =='April',
            data_Finance_v3.Month =='May',
            data_Finance_v3.Month =='June',
            data_Finance_v3.Month =='July',
            data_Finance_v3.Month =='August',
            data_Finance_v3.Month =='September',
            data_Finance_v3.Month =='October',
            data_Finance_v3.Month =='November',
            data_Finance_v3.Month =='December'

            ]
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Finance_v3['Quarter']=np.select(cond,val)

            data_Finance_vq=data_Finance_v3.groupby('Quarter').agg({'EBIDTA':'mean'}).reset_index()
            data_Finance_vq1=data_Finance_vq[data_Finance_vq.EBIDTA==data_Finance_vq.EBIDTA.max()]
            x=data_Finance_vq1['Quarter'].sum()
            y=round(data_Finance_vq1['EBIDTA'].sum(),0)
            y_view=f"{y:,}"
            output9 = str("5. The property had the EBIDTA in {} with {}.".format(x,y_view))
            
            
            # Month over Month
            data_Finance_v3=Sort_Dataframeby_Month(data_Finance_v3,monthcolumnname='Month')
            data_Finance_v3=data_Finance_v3.reset_index()
            data_Finance_v3['index_1'] = data_Finance_v3['index']+1
            data_Finance_v4 = pd.merge(data_Finance_v3[['Month','EBIDTA','index']],data_Finance_v3[['Month','EBIDTA','index_1']],left_on='index',right_on='index_1',how='left')
            data_Finance_v4.rename({'Month_x':'Month','EBIDTA_x':'CM','EBIDTA_y':'LM'},axis=1,inplace=True)
            data_Finance_v4=data_Finance_v4[['Month','CM','LM','index']]
            data_Finance_v4['LM']=np.where(data_Finance_v4.LM.isnull(),data_Finance_v4.CM,data_Finance_v4.LM)
            data_Finance_v4['MOM'] = data_Finance_v4['CM']-data_Finance_v4['LM']
            data_Finance_vx=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.min()]
            data_Finance_vy=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.max()]
            
            a=data_Finance_vx.Month.sum()
            b=round(abs(data_Finance_vx.MOM.sum()),0)
            b=f"{b:,}"
            c=data_Finance_vx['index'].sum()
            d=data_Finance_v4[data_Finance_v4['index']==(c-1)].Month.sum()

            e=data_Finance_vy.Month.sum()
            f=round(abs(data_Finance_vy.MOM.sum()),0)
            f=f"{f:,}"
            g=data_Finance_vy['index'].sum()
            h=data_Finance_v4[data_Finance_v4['index']==(g-1)].Month.sum()
            
            data = [go.Scatter(x=data_Finance_v3['Month'],y=data_Finance_v3['EBIDTA'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            fig.update_layout(title_text='EBIDTA Vs Month', title_x=0.5)
            img = plot(fig,filename='ebidta.html',config={'displayModeBar':True}, output_type = 'div')


            output10=str("6. The highest Month over Month decrease in EBIDTA occured from {} to {}, where the value fell by {}. The property needs to check on why there is a sudden dip.".format(d,a,b))
            output11=str("7. The highest Month over Month increase in EBIDTA occured from {} to {}, where the value grew by {}. The property has the highest surge of cash flow here.".format(h,e,f))
            
            fin1 = output1+output2
            fin2 = os.linesep+os.linesep+output3+os.linesep+os.linesep+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8+os.linesep+os.linesep+output9+os.linesep+os.linesep+output10+os.linesep+os.linesep+output11
            img_text = str("I have created a trend chart for the performance indicator - EBIDTA, it is open and available in your browser for viewing")

            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img) 

    @asyncio.coroutine
    async def goppar_func(self,hotel,month,turn_context:TurnContext): # GOP per available room =  GOP/Available Rooms
        
        hotel_list = data_Hotel_fin.Hotel.unique().tolist() 
        
        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')
        if hotel.lower().replace(' ','') not in hotel_list:
            print('\nThe available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            goppar_func()
        else:
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_Finance_v1 = data_Finance[data_Finance['Month']==month]
            data_rooms_v1 = data_Hotel[data_Hotel['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            
            revenue_c = data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].CY.sum()
            expenses_c = data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].CY.sum()
            
            gop_c = revenue_c-expenses_c
            total_rooms = data_rooms_v1['Total Rooms'].sum()
            goppar_c = round((gop_c/total_rooms),0)
            output1 = str("The Gross Operating Profit Per Available Room for {} in current fiscal year for the month of {} is ${}".format(hotel,month,goppar_c))
            
            
            # INSIGHTS
            
            # OVERALL
            
            # Best hotel by GOPPAR
            data_Finance_v2 = data_Finance.copy()
            revenue=data_Finance_v2[data_Finance_v2['Category']=='Total Revenue'].groupby('Hotel').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v2[data_Finance_v2['Category']=='Total Expenses'].groupby('Hotel').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v2[data_Finance_v2['Category']=='Total Expense'].groupby('Hotel').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Hotel')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Hotel')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3=pd.merge(data_Finance_v3,data_Hotel,on='Hotel')
            data_Finance_v3['GOPPAR']=round(data_Finance_v3['GOP']/data_Finance_v3['Total Rooms'],2)
            data_Finance_v3=data_Finance_v3[['Hotel','GOPPAR']]
            
            goppar_max=data_Finance_v3[data_Finance_v3.GOPPAR == data_Finance_v3.GOPPAR.max()]
            goppar_cur = data_Finance_v3[data_Finance_v3['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=goppar_max.Hotel.sum()
            j=goppar_max.GOPPAR.sum()
            i=f"{j:,}"
            u=goppar_cur.Hotel.sum()
            m=goppar_cur.GOPPAR.sum()
            k=round(j-m,0)
            k=f"{k:,}"
            m_view = f"{m:,}"
            
            output2 = str("I have collated some interesting insights about the property {} -".format(hotel))

            if q==u:
                output3=str("1. The property {} you are interested in is the best performing hotel, with GOPPAR of {}".format(q,i))
            else:
                if m >= 0:
                    output3=str("1. The best performing property is {}. {} has {} less GOPPAR compared to the best performing hotel".format(q,hotel,k))
                else:
                    output3=str("1. The best performing property is {} with GOPPAR {}, while {} has GOPPAR {}, meaning poor profit generated per room.".format(q,i,hotel,m_view))
            
            
            # Hotel wise insights - LY and same month
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_Finance_vly = data_Finance_v1[data_Finance_v1['Month']==month]
            revenue_l = data_Finance_vly[data_Finance_vly['Category']=='Total Revenue'].LY.sum()
            expenses_l = data_Finance_vly[data_Finance_vly['Category']=='Total Expenses'].LY.sum()
            gop_l = round(revenue_l-expenses_l,0)
            goppar_l = round(gop_l/total_rooms,0)
            goppar_c_view=f"{goppar_c:,}"
            goppar_l_view=f"{goppar_l:,}"
            
            if  goppar_c>goppar_l:
                output4=str("2. There is an increase in GOPPAR from last year to this year in {} in {} from {} to {}.".format(hotel,month,goppar_l_view,goppar_c_view))
            elif goppar_c<goppar_l:
                output4=str("2. There is a decrease in GOPPAR from last year to this year in {} in {} from {} to {}.".format(hotel,month,goppar_l_view,goppar_c_view))
            else:
                output4=str("2. There is no change in GOPPAR from last year to this year in {} in {} with GOPPAR being {}.".format(hotel,month,goppar_l_view))
            
            
            
            # GOPPAR in the LY - Overall
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue_lo = data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].LY.sum()
            expenses_lo = data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].LY.sum()
            gop_lo = revenue_lo-expenses_lo
            goppar_lo = round(gop_lo/total_rooms,0)
            goppar_lo_view=f"{goppar_lo:,}"
            
            if  m>goppar_lo:
                output5=str("3. There is an increase in GOPPAR from last year to this year in {} from {} to {}.".format(hotel,goppar_lo_view,m_view))
            elif m<goppar_lo:
                output5=str("3. There is a decrease in GOPPAR from last year to this year in {} from {} to {}.".format(hotel,goppar_lo_view,m_view))
            else:
                output5=str("3. There is no change in GOPPAR from last year to this year in {} with GOPPAR being {}.".format(hotel,goppar_lo_view))
            
            
            # Month with highest GOPPAR
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue=data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].groupby('Month').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].groupby('Month').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v1[data_Finance_v1['Category']=='Total Expense'].groupby('Month').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Month')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Month')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3['GOPPAR']=round(data_Finance_v3['GOP']/total_rooms,0)
            data_Finance_v3=data_Finance_v3[['Month','GOPPAR']]
            data_Finance_v4=data_Finance_v3[data_Finance_v3.GOPPAR==data_Finance_v3.GOPPAR.max()]
            y=data_Finance_v4.GOPPAR.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output6 = str("4. The property had the best GOPPAR of {} in month {}, making it the month with highest gross profit obtained.".format(y,x))

            # Month with lowest GOPPAR
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue=data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].groupby('Month').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].groupby('Month').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v1[data_Finance_v1['Category']=='Total Expense'].groupby('Month').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Month')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Month')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3['GOPPAR']=round(data_Finance_v3['GOP']/total_rooms,0)
            data_Finance_v3=data_Finance_v3[['Month','GOPPAR']]
            data_Finance_v4=data_Finance_v3[data_Finance_v3.GOPPAR==data_Finance_v3.GOPPAR.min()]
            y=data_Finance_v4.GOPPAR.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output7 = str("5. The property had the lowest GOPPAR of {} in month {}, making it the month with lowest gross profit obtained.".format(y,x))
            
            
            # Quarter with best performance
            cond = [data_Finance_v3.Month =='January',
            data_Finance_v3.Month =='February',
            data_Finance_v3.Month =='March',
            data_Finance_v3.Month =='April',
            data_Finance_v3.Month =='May',
            data_Finance_v3.Month =='June',
            data_Finance_v3.Month =='July',
            data_Finance_v3.Month =='August',
            data_Finance_v3.Month =='September',
            data_Finance_v3.Month =='October',
            data_Finance_v3.Month =='November',
            data_Finance_v3.Month =='December'

            ]
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Finance_v3['Quarter']=np.select(cond,val)

            data_Finance_vq=data_Finance_v3.groupby('Quarter').agg({'GOPPAR':'mean'}).reset_index()
            data_Finance_vq1=data_Finance_vq[data_Finance_vq.GOPPAR==data_Finance_vq.GOPPAR.max()]
            x=data_Finance_vq1['Quarter'].sum()
            y=round(data_Finance_vq1['GOPPAR'].sum(),0)
            y_view=f"{y:,}"
            output8 = str("6. The property had the GOPPAR in {} with {}.".format(x,y_view))
            
            
            
            # Month over Month
            data_Finance_v3=Sort_Dataframeby_Month(data_Finance_v3,monthcolumnname='Month')
            data_Finance_v3=data_Finance_v3.reset_index()
            data_Finance_v3['index_1'] = data_Finance_v3['index']+1
            data_Finance_v4 = pd.merge(data_Finance_v3[['Month','GOPPAR','index']],data_Finance_v3[['Month','GOPPAR','index_1']],left_on='index',right_on='index_1',how='left')
            data_Finance_v4.rename({'Month_x':'Month','GOPPAR_x':'CM','GOPPAR_y':'LM'},axis=1,inplace=True)
            data_Finance_v4=data_Finance_v4[['Month','CM','LM','index']]
            data_Finance_v4['LM']=np.where(data_Finance_v4.LM.isnull(),data_Finance_v4.CM,data_Finance_v4.LM)
            data_Finance_v4['MOM'] = data_Finance_v4['CM']-data_Finance_v4['LM']
            data_Finance_vx=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.min()]
            data_Finance_vy=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.max()]
            
            a=data_Finance_vx.Month.sum()
            b=round(abs(data_Finance_vx.MOM.sum()),0)
            b=f"{b:,}"
            c=data_Finance_vx['index'].sum()
            d=data_Finance_v4[data_Finance_v4['index']==(c-1)].Month.sum()

            e=data_Finance_vy.Month.sum()
            f=round(abs(data_Finance_vy.MOM.sum()),0)
            f=f"{f:,}"
            g=data_Finance_vy['index'].sum()
            h=data_Finance_v4[data_Finance_v4['index']==(g-1)].Month.sum()
            
            data = [go.Scatter(x=data_Finance_v3['Month'],y=data_Finance_v3['GOPPAR'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            fig.update_layout(title_text='GOPPAR Vs Month', title_x=0.5)
            img = plot(fig,filename='goppar.html',config={'displayModeBar':True}, output_type = 'div')


            output9=str("7. The highest Month over Month decrease in GOPPAR occured from {} to {}, where the value fell by {}. The property needs to check on why there is a sudden dip.".format(d,a,b))
            output10=str("8. The highest Month over Month increase in GOPPAR occured from {} to {}, where the value grew by {}. The property has the highest increase in gross profit here.".format(h,e,f))

            fin1 = output1
            fin2 = os.linesep+output2+os.linesep+os.linesep+os.linesep+output3+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8+os.linesep+os.linesep+output9+os.linesep+os.linesep+output10
            img_text = str("I have created a trend chart for the performance indicator - GOPPAR, it is open and available in your browser for viewing")
            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img)

                
    @asyncio.coroutine
    async def gop_func(self,hotel,month,turn_context:TurnContext): # Gross Operating Profit = Total Revenue - Total Expenses
        
        
        hotel_list = data_Hotel_fin.Hotel.unique().tolist() 
        
        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')
        if hotel.lower().replace(' ','') not in hotel_list:
            print('\nThe available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            gop_func()
        else:
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_Finance_v1 = data_Finance[data_Finance['Month']==month]
            data_rooms_v1 = data_Hotel[data_Hotel['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue_c = data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].CY.sum()
            expenses_c = data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].CY.sum()
            gop_c = round(revenue_c-expenses_c,0)
            gop_c_view = f"{gop_c:,}"
            output1 = str("The Gross Operating Profit for {} in current fiscal year for the month of {} is ${}".format(hotel,month,gop_c_view))
            
            
            # Overall Insights
            
            # Hotel comparison
            
            data_Finance_v2 = data_Finance.copy()
            revenue=data_Finance_v2[data_Finance_v2['Category']=='Total Revenue'].groupby('Hotel').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v2[data_Finance_v2['Category']=='Total Expenses'].groupby('Hotel').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v2[data_Finance_v2['Category']=='Total Expense'].groupby('Hotel').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Hotel')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Hotel')
            data_Finance_v3['GOP'] = round(data_Finance_v3['Revenue'] - data_Finance_v3['Expenses'],0)
            data_Finance_v3=data_Finance_v3[['Hotel','GOP']]
            
            gop_max=data_Finance_v3[data_Finance_v3.GOP == data_Finance_v3.GOP.max()]
            gop_cur = data_Finance_v3[data_Finance_v3['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=gop_max.Hotel.sum()
            j=gop_max.GOP.sum()
            i=f"{j:,}"
            u=gop_cur.Hotel.sum()
            m=gop_cur.GOP.sum()
            k=round(j-m,0)
            k=f"{k:,}"
            m_view = f"{m:,}"
            
            output2 = str("I have collated some interesting insights about the property {} -".format(hotel))

            if q==u:
                output3=str("1. The property {} you are interested in is the best performing hotel, with GOPPAR of {}".format(q,i))
            else:
                if m >= 0:
                    output3=str("1. The best performing property is {}. {} has {} less GOPPAR compared to the best performing hotel".format(q,hotel,k))
                else:
                    output3=str("1. The best performing property is {} with GOPPAR {}, while {} has GOPPAR {}, meaning poor profit generated.".format(q,i,hotel,m_view))
                    
             # Hotel wise insights - LY and same month
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_Finance_vly = data_Finance_v1[data_Finance_v1['Month']==month]
            revenue_l = data_Finance_vly[data_Finance_vly['Category']=='Total Revenue'].LY.sum()
            expenses_l = data_Finance_vly[data_Finance_vly['Category']=='Total Expenses'].LY.sum()
            gop_l = round(revenue_l-expenses_l,0)
            gop_c_view=f"{gop_c:,}"
            gop_l_view=f"{gop_l:,}"
            
            if  gop_c>gop_l:
                output4=str("2. There is an increase in GOP from last year to this year in {} in {} from {} to {}.".format(hotel,month,gop_l_view,gop_c_view))
            elif gop_c<gop_l:
                output4=str("2. There is a decrease in GOP from last year to this year in {} in {} from {} to {}.".format(hotel,month,gop_l_view,gop_c_view))
            else:
                output4=str("2. There is no change in GOP from last year to this year in {} in {} with GOP being {}.".format(hotel,month,gop_l_view))
            
            
            
            # GOP in the LY - Overall
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue_lo = data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].LY.sum()
            expenses_lo = data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].LY.sum()
            gop_lo = revenue_lo-expenses_lo
            gop_lo_view=f"{gop_lo:,}"
            
            if  m>gop_lo:
                output5=str("3. There is an increase in GOP from last year to this year in {} from {} to {}.".format(hotel,gop_lo_view,m_view))
            elif m<gop_lo:
                output5=str("3. There is a decrease in GOP from last year to this year in {} from {} to {}.".format(hotel,gop_lo_view,m_view))
            else:
                output5=str("3. There is no change in GOP from last year to this year in {} with GOPPAR being {}.".format(hotel,gop_lo_view))
            
            
            # Month with highest GOP
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue=data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].groupby('Month').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].groupby('Month').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v1[data_Finance_v1['Category']=='Total Expense'].groupby('Month').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Month')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Month')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3=data_Finance_v3[['Month','GOP']]
            data_Finance_v4=data_Finance_v3[data_Finance_v3.GOP==data_Finance_v3.GOP.max()]
            y=data_Finance_v4.GOP.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output6 = str("4. The property had the best GOP of {} in month {}, making it the month with highest gross profit obtained.".format(y,x))

            # Month with lowest GOP
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            revenue=data_Finance_v1[data_Finance_v1['Category']=='Total Revenue'].groupby('Month').CY.sum().reset_index()
            revenue.rename({'CY':'Revenue'},axis=1,inplace=True)
            expenses=data_Finance_v1[data_Finance_v1['Category']=='Total Expenses'].groupby('Month').CY.sum().reset_index()
            expenses.rename({'CY':'Expenses'},axis=1,inplace=True)
            expense=data_Finance_v1[data_Finance_v1['Category']=='Total Expense'].groupby('Month').CY.sum().reset_index()
            expense.rename({'CY':'Expense'},axis=1,inplace=True)
            data_Finance_v3 = pd.merge(revenue,expenses,on='Month')
            data_Finance_v3 = pd.merge(data_Finance_v3,expense,on='Month')
            data_Finance_v3['GOP'] = data_Finance_v3['Revenue'] - data_Finance_v3['Expenses']
            data_Finance_v3=data_Finance_v3[['Month','GOP']]
            data_Finance_v4=data_Finance_v3[data_Finance_v3.GOP==data_Finance_v3.GOP.min()]
            y=data_Finance_v4.GOP.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output7 = str("5. The property had the lowest GOP of {} in month {}, making it the month with lowest gross profit obtained.".format(y,x))
            
            
            # Quarter with best performance
            cond = [data_Finance_v3.Month =='January',
            data_Finance_v3.Month =='February',
            data_Finance_v3.Month =='March',
            data_Finance_v3.Month =='April',
            data_Finance_v3.Month =='May',
            data_Finance_v3.Month =='June',
            data_Finance_v3.Month =='July',
            data_Finance_v3.Month =='August',
            data_Finance_v3.Month =='September',
            data_Finance_v3.Month =='October',
            data_Finance_v3.Month =='November',
            data_Finance_v3.Month =='December'

            ]
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Finance_v3['Quarter']=np.select(cond,val)

            data_Finance_vq=data_Finance_v3.groupby('Quarter').agg({'GOP':'mean'}).reset_index()
            data_Finance_vq1=data_Finance_vq[data_Finance_vq.GOP==data_Finance_vq.GOP.max()]
            x=data_Finance_vq1['Quarter'].sum()
            y=round(data_Finance_vq1['GOP'].sum(),0)
            y_view=f"{y:,}"
            output8 = str("6. The property had the best GOP in {} with {}.".format(x,y_view))
            
            
            
            # Month over Month
            data_Finance_v3=Sort_Dataframeby_Month(data_Finance_v3,monthcolumnname='Month')
            data_Finance_v3=data_Finance_v3.reset_index()
            data_Finance_v3['index_1'] = data_Finance_v3['index']+1
            data_Finance_v4 = pd.merge(data_Finance_v3[['Month','GOP','index']],data_Finance_v3[['Month','GOP','index_1']],left_on='index',right_on='index_1',how='left')
            data_Finance_v4.rename({'Month_x':'Month','GOP_x':'CM','GOP_y':'LM'},axis=1,inplace=True)
            data_Finance_v4=data_Finance_v4[['Month','CM','LM','index']]
            data_Finance_v4['LM']=np.where(data_Finance_v4.LM.isnull(),data_Finance_v4.CM,data_Finance_v4.LM)
            data_Finance_v4['MOM'] = data_Finance_v4['CM']-data_Finance_v4['LM']
            data_Finance_vx=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.min()]
            data_Finance_vy=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.max()]
            
            a=data_Finance_vx.Month.sum()
            b=round(abs(data_Finance_vx.MOM.sum()),0)
            b=f"{b:,}"
            c=data_Finance_vx['index'].sum()
            d=data_Finance_v4[data_Finance_v4['index']==(c-1)].Month.sum()

            e=data_Finance_vy.Month.sum()
            f=round(abs(data_Finance_vy.MOM.sum()),0)
            f=f"{f:,}"
            g=data_Finance_vy['index'].sum()
            h=data_Finance_v4[data_Finance_v4['index']==(g-1)].Month.sum()
            
            data = [go.Scatter(x=data_Finance_v3['Month'],y=data_Finance_v3['GOP'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            fig.update_layout(title_text='GOP Vs Month', title_x=0.5)
            img = plot(fig,filename='gop.html',config={'displayModeBar':True}, output_type = 'div')


            output9=str("7. The highest Month over Month decrease in GOP occured from {} to {}, where the value fell by {}. The property needs to check on why there is a sudden dip.".format(d,a,b))
            output10=str("8. The highest Month over Month increase in GOP occured from {} to {}, where the value grew by {}. The property has the highest increase in gross profit here.".format(h,e,f))
            
            fin1 = output1
            fin2 = os.linesep+output2+os.linesep+os.linesep+os.linesep+output3+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8+os.linesep+os.linesep+output9+os.linesep+os.linesep+output10
            img_text = str("A trend chart for the performance indicator - GOP is open and available in your browser for viewing.")
            
            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img)            
            
    
    @asyncio.coroutine
    async def alos_func(self,hotel,month,turn_context:TurnContext): # Average Length of stay (ALOS) = Total occupied room nights / number of bookings
        

        hotel_list = data_Hotel_fin.Hotel.unique().tolist() 
        
        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')
            
        if hotel.lower().replace(' ','') not in hotel_list:
            
            print('\nThe available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            alos_func()
        else:
            
            data_Hotel_1['ALOS_Calc'] = data_Hotel_1['Total Room Nights (Length of Stay)']/data_Hotel_1['Room Sold']
            data_Hotel_2 = data_Hotel_1[data_Hotel_1.Month==month]
            alos_calc = data_Hotel_2[data_Hotel_2['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')].ALOS_Calc.mean()
            output1 = str("The Average Length of Stay at {} in {} is {} days".format(hotel,month,round(alos_calc,0)))
            
            # Best hotel and comparison with your hotel

            data_Hotel_fin_w1=data_Hotel_1.groupby('Hotel').agg({'Total Room Nights (Length of Stay)':'sum','Room Sold':'sum'}).reset_index()
            data_Hotel_fin_w1['ALOS']=round((data_Hotel_fin_w1['Total Room Nights (Length of Stay)']/data_Hotel_fin_w1['Room Sold']),2)
            data_Hotel_fin_w1=data_Hotel_fin_w1[['Hotel','ALOS']]
            data_Hotel_fin_w2=data_Hotel_fin_w1[data_Hotel_fin_w1.ALOS == data_Hotel_fin_w1.ALOS.max()]
            data_Hotel_fin_w3=data_Hotel_fin_w1[data_Hotel_fin_w1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=data_Hotel_fin_w2.Hotel.sum()
            j=data_Hotel_fin_w2.ALOS.sum()
            i=str(j)+' days'
            u=data_Hotel_fin_w3.Hotel.sum()
            m=data_Hotel_fin_w3.ALOS.sum()
            k=round(j-m,1)
            k=str(k)+' days'
            
            output2=str("I have collated some interesting insights about the property {} - ".format(hotel))

            if q==u:
                output3=str("1. The property {} you are interested in is the best performing hotel, with an ALOS of {}".format(q,i))
            else:
                output3=str("1. The best performing property is: {}. {} has {} less ALOS compared to the best performing hotel".format(q,hotel,k))


            # Hotel wise Insights

            # Best performing quarter

            data_Hotel_fin_v1 = data_Hotel_1[(data_Hotel_1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            cond = [data_Hotel_fin_v1.Month =='January',
            data_Hotel_fin_v1.Month =='February',
            data_Hotel_fin_v1.Month =='March',
            data_Hotel_fin_v1.Month =='April',
            data_Hotel_fin_v1.Month =='May',
            data_Hotel_fin_v1.Month =='June',
            data_Hotel_fin_v1.Month =='July',
            data_Hotel_fin_v1.Month =='August',
            data_Hotel_fin_v1.Month =='September',
            data_Hotel_fin_v1.Month =='October',
            data_Hotel_fin_v1.Month =='November',
            data_Hotel_fin_v1.Month =='December'
            ]
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Hotel_fin_v1['Quarter']=np.select(cond,val)

            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Quarter').agg({'Total Room Nights (Length of Stay)':'sum','Room Sold':'sum'}).reset_index()
            data_Hotel_fin_v2['ALOS']=round((data_Hotel_fin_v2['Total Room Nights (Length of Stay)']/data_Hotel_fin_v2['Room Sold']),2)
            data_Hotel_fin_v3=data_Hotel_fin_v2[data_Hotel_fin_v2.ALOS==data_Hotel_fin_v2.ALOS.max()]
            x=data_Hotel_fin_v3['Quarter'].sum()
            output4 = str("2. The property had the best performance in {}".format(x))

            # Best Performing month

            data_Hotel_fin_v1 = data_Hotel_1[(data_Hotel_1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Month').agg({'Total Room Nights (Length of Stay)':'sum','Room Sold':'sum'}).reset_index()
            data_Hotel_fin_v2['ALOS']=round((data_Hotel_fin_v2['Total Room Nights (Length of Stay)']/data_Hotel_fin_v2['Room Sold']),2)
            data_Hotel_fin_v3=data_Hotel_fin_v2[data_Hotel_fin_v2.ALOS==data_Hotel_fin_v2.ALOS.max()]
            x=data_Hotel_fin_v3['Month'].sum()
            x_mean = data_Hotel_fin_v2['ALOS'].mean()
            data_Hotel_fin_vx=data_Hotel_fin_v2[data_Hotel_fin_v2.Month==x]
            diff = round((data_Hotel_fin_vx.ALOS.sum()-x_mean),1)
            diff = str(diff) + ' days'
            output5 = str("3. The property had the best ALOS in month {}, making it the month where customer stayed the longest. The hotel had {} higher occupancy than average.".format(x,diff))

            # Worst Performing month

            data_Hotel_fin_v1 = data_Hotel_1[(data_Hotel_1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Month').agg({'Total Room Nights (Length of Stay)':'sum','Room Sold':'sum'}).reset_index()
            data_Hotel_fin_v2['ALOS']=round((data_Hotel_fin_v2['Total Room Nights (Length of Stay)']/data_Hotel_fin_v2['Room Sold']),2)
            data_Hotel_fin_v3=data_Hotel_fin_v2[data_Hotel_fin_v2.ALOS==data_Hotel_fin_v2.ALOS.min()]
            x=data_Hotel_fin_v3['Month'].sum()
            x_mean = data_Hotel_fin_v2['ALOS'].mean()
            data_Hotel_fin_vx=data_Hotel_fin_v2[data_Hotel_fin_v2.Month==x]
            diff = round((x_mean-data_Hotel_fin_vx.ALOS.sum()),1)
            diff = str(diff) + ' days'
            output6 = str("4. The property had the worst ALOS in month {}, making it the month where customer stayed the least days. The hotel had {} lower occupancy than average.".format(x,diff))

            # Month Over Month highest decrease and increase
            data_Hotel_fin_v1 = data_Hotel_1[(data_Hotel_1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ',''))]
            data_Hotel_fin_v2=data_Hotel_fin_v1.groupby('Month').agg({'Total Room Nights (Length of Stay)':'sum','Room Sold':'sum'}).reset_index()
            data_Hotel_fin_v2['ALOS']=round((data_Hotel_fin_v2['Total Room Nights (Length of Stay)']/data_Hotel_fin_v2['Room Sold']),2)
            data_Hotel_fin_v2=Sort_Dataframeby_Month(data_Hotel_fin_v2,monthcolumnname='Month')
            data_Hotel_fin_v2=data_Hotel_fin_v2.reset_index()
            data_Hotel_fin_v2['index_1'] = data_Hotel_fin_v2['index']+1
            data_Hotel_fin_v3 = pd.merge(data_Hotel_fin_v2[['Month','ALOS','index']],data_Hotel_fin_v2[['Month','ALOS','index_1']],left_on='index',right_on='index_1',how='left')
            data_Hotel_fin_v3.rename({'Month_x':'Month','ALOS_x':'CM','ALOS_y':'LM'},axis=1,inplace=True)
            data_Hotel_fin_v3=data_Hotel_fin_v3[['Month','CM','LM','index']]
            data_Hotel_fin_v3['LM']=np.where(data_Hotel_fin_v3.LM.isnull(),data_Hotel_fin_v3.CM,data_Hotel_fin_v3.LM)
            data_Hotel_fin_v3['MOM'] = data_Hotel_fin_v3['CM']-data_Hotel_fin_v3['LM']
            data_Hotel_fin_vx=data_Hotel_fin_v3[data_Hotel_fin_v3.MOM==data_Hotel_fin_v3.MOM.min()]
            data_Hotel_fin_vy=data_Hotel_fin_v3[data_Hotel_fin_v3.MOM==data_Hotel_fin_v3.MOM.max()]
            a=data_Hotel_fin_vx.Month.sum()
            b=round(abs(data_Hotel_fin_vx.MOM.sum()),1)
            b=str(b)+' days'
            c=data_Hotel_fin_vx['index'].sum()
            d=data_Hotel_fin_v3[data_Hotel_fin_v3['index']==(c-1)].Month.sum()

            e=data_Hotel_fin_vy.Month.sum()
            f=round(abs(data_Hotel_fin_vy.MOM.sum()),1)
            f=str(f)+' days'
            g=data_Hotel_fin_vy['index'].sum()
            h=data_Hotel_fin_v3[data_Hotel_fin_v3['index']==(g-1)].Month.sum()
            
            data = [go.Scatter(x=data_Hotel_fin_v2['Month'],y=data_Hotel_fin_v2['ALOS'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            fig.update_layout(title_text='ALOS Vs Month', title_x=0.5)
            img = plot(fig,filename='occrate.html',config={'displayModeBar':True}, output_type = 'div')


            output7=str("5. The highest Month over Month decrease in ALOS occured from {} to {}, where the value fell by {}. The property needs to check up on why there is a sudden dip.".format(d,a,b))
            output8=str("6. The highest Month over Month increase in ALOS occured from {} to {}, where the value grew by {}. The property needs to prepare themselves as customers are expected to stay longer during this period.".format(h,e,f))

            fin1 = output1+os.linesep
            fin2 = output2+os.linesep+os.linesep+os.linesep+output3+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8
            img_text = str("A trend chart for the performance indicator - ALOS is open and available in your browser for viewing.") 
            
            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img)
        
    @asyncio.coroutine
    async def adr_func(self,hotel,month,turn_context:TurnContext): # Average Daily Rate = Room Revenue/Number of rooms sold or occupied

        hotel_list = data_Hotel_fin.Hotel.unique().tolist() 

        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')

        if hotel.lower().replace(' ','') not in hotel_list:
            print('\nThe available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            adr_func()   
        else:       
            data_Finance_v1 = data_Finance[data_Finance['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_rooms_v1 = data_Hotel_1[data_Hotel_1['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            df = pd.merge(data_Finance_v1[data_Finance_v1['Sub Category']=='Room Revenue'][['Month','CY']],data_rooms_v1[['Month','Room Sold']], on = 'Month',how = 'left')
            subdf = df.groupby('Month').agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            subdf = subdf[subdf.Month==month]
            num = subdf.CY.sum()
            den = subdf['Room Sold'].sum()
            adr_mon = round(num/den,0)
            adr_mon_view = f"{adr_mon:,}"
            
            output1 = str("The Average Daily Rate for {} in the month {} in the current year is {}.".format(hotel,month,adr_mon_view))
            
            # Insights
            
            # Overall
            
            output2=str("I have collated some interesting insights about the property {} -".format(hotel))
            
            df = pd.merge(data_Finance[data_Finance['Sub Category']=='Room Revenue'][['Hotel','Month','CY','LY']],data_Hotel_1[['Hotel','Month','Room Sold']], on = ['Hotel','Month'],how = 'left')
            subdf = df.groupby('Hotel').agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Hotel','adr']]
            data_Hotel_fin_w2=data_Hotel_fin_w1[data_Hotel_fin_w1.adr == data_Hotel_fin_w1.adr.max()]
            data_Hotel_fin_w3=data_Hotel_fin_w1[data_Hotel_fin_w1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=data_Hotel_fin_w2.Hotel.sum()
            j=data_Hotel_fin_w2.adr.sum()
            j_view = f"{j:,}"
            i=str(j_view)+' $'
            u=data_Hotel_fin_w3.Hotel.sum()
            m=data_Hotel_fin_w3.adr.sum()
            m_view = f"{m:,}"
            k=round(j-m,0)
            k_view = f"{k:,}"
            k=str(k_view)+' $'

            if q==u:
                output3=str("1. The property {} you are interested in is the best performing hotel, with an ADR of {}".format(q,i))
            else:
                output3=str("1. The best performing property is: {}. {} has {} less ADR compared to the best performing hotel".format(q,hotel,k))

            # Hotel wise insights
            
            # Hotel wise insights - LY and same month
            data_Finance_v1 = df[df['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_Finance_vly = data_Finance_v1[data_Finance_v1['Month']==month]
            subdf = data_Finance_vly.groupby('Hotel').agg({'LY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['LY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Hotel','adr']]
            
            adr_l = data_Hotel_fin_w1.adr.sum()
            adr_l_view=f"{adr_l:,}"
            
            if  adr_mon>adr_l:
                output4=str("2. There is an increase in ADR from last year to this year in {} in {} from {} to {}.".format(hotel,month,adr_l_view,adr_mon_view))
            elif adr_mon<adr_l:
                output4=str("2. There is a decrease in ADR from last year to this year in {} in {} from {} to {}.".format(hotel,month,adr_l_view,adr_mon_view))
            else:
                output4=str("2. There is no change in ADR from last year to this year in {} in {} with GOP being {}.".format(hotel,month,adr_l_view))
            
            
             # ADR in the LY - Overall
            subdf = data_Finance_v1.groupby('Hotel').agg({'LY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['LY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Hotel','adr']]
            adr_lo = data_Hotel_fin_w1.adr.sum()
            adr_lo_view=f"{adr_lo:,}"
            
            if  m>adr_lo:
                output5=str("3. There is an increase in ADR from last year to this year in {} from {} to {}.".format(hotel,adr_lo_view,m_view))
            elif m<adr_lo:
                output5=str("3. There is a decrease in ADR from last year to this year in {} from {} to {}.".format(hotel,adr_lo_view,m_view))
            else:
                output5=str("3. There is no change in ADR from last year to this year in {} with ADR being {}.".format(hotel,adr_lo_view))
            
            
            # Month with highest ADR
            data_Finance_v1 = df[df['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            subdf = data_Finance_v1.groupby('Month').agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Month','adr']]
            data_Finance_v4=data_Hotel_fin_w1[data_Hotel_fin_w1.adr==data_Hotel_fin_w1.adr.max()]
            y=data_Finance_v4.adr.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output6 = str("4. The property had the best ADR of {} in month {}, making it the month with highest daily revenue.".format(y,x))
            
            
            # Month with lowest ADR
            data_Finance_v1 = df[df['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            subdf = data_Finance_v1.groupby('Month').agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Month','adr']]
            data_Finance_v4=data_Hotel_fin_w1[data_Hotel_fin_w1.adr==data_Hotel_fin_w1.adr.min()]
            y=data_Finance_v4.adr.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output7 = str("5. The property had the lowest ADR of {} in month {}, making it the month with lowest daily revenue.".format(y,x))

            # Quarter with best performance
            data_Finance_v3 = df[df['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            cond = [data_Finance_v3.Month =='January',
            data_Finance_v3.Month =='February',
            data_Finance_v3.Month =='March',
            data_Finance_v3.Month =='April',
            data_Finance_v3.Month =='May',
            data_Finance_v3.Month =='June',
            data_Finance_v3.Month =='July',
            data_Finance_v3.Month =='August',
            data_Finance_v3.Month =='September',
            data_Finance_v3.Month =='October',
            data_Finance_v3.Month =='November',
            data_Finance_v3.Month =='December'

            ]
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Finance_v3['Quarter']=np.select(cond,val)
            subdf = data_Finance_v3.groupby('Quarter').agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Quarter','adr']]
            data_Finance_vq1=data_Hotel_fin_w1[data_Hotel_fin_w1.adr==data_Hotel_fin_w1.adr.max()]
            x=data_Finance_vq1['Quarter'].sum()
            y=round(data_Finance_vq1['adr'].sum(),0)
            y_view=f"{y:,}"
            output8 = str("6. The property had the best ADR in {} with {}.".format(x,y_view))
            
            
            
            # Month over Month
            data_Finance_v1 = df[df['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            subdf = data_Finance_v1.groupby('Month').agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            data_Hotel_fin_w1=subdf[['Month','adr']]
            
            
            data_Finance_v3=Sort_Dataframeby_Month(data_Hotel_fin_w1,monthcolumnname='Month')
            data_Finance_v3=data_Finance_v3.reset_index()
            data_Finance_v3['index_1'] = data_Finance_v3['index']+1
            data_Finance_v4 = pd.merge(data_Finance_v3[['Month','adr','index']],data_Finance_v3[['Month','adr','index_1']],left_on='index',right_on='index_1',how='left')
            data_Finance_v4.rename({'Month_x':'Month','adr_x':'CM','adr_y':'LM'},axis=1,inplace=True)
            data_Finance_v4=data_Finance_v4[['Month','CM','LM','index']]
            data_Finance_v4['LM']=np.where(data_Finance_v4.LM.isnull(),data_Finance_v4.CM,data_Finance_v4.LM)
            data_Finance_v4['MOM'] = data_Finance_v4['CM']-data_Finance_v4['LM']
            data_Finance_vx=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.min()]
            data_Finance_vy=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.max()]
            
            a=data_Finance_vx.Month.sum()
            b=round(abs(data_Finance_vx.MOM.sum()),2)
            b=f"{b:,}"
            c=data_Finance_vx['index'].sum()
            d=data_Finance_v4[data_Finance_v4['index']==(c-1)].Month.sum()

            e=data_Finance_vy.Month.sum()
            f=round(abs(data_Finance_vy.MOM.sum()),0)
            f=f"{f:,}"
            g=data_Finance_vy['index'].sum()
            h=data_Finance_v4[data_Finance_v4['index']==(g-1)].Month.sum()
            
            data = [go.Scatter(x=data_Finance_v3['Month'],y=data_Finance_v3['adr'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            fig.update_layout(title_text='ADR Vs Month', title_x=0.5)
            img = plot(fig,filename='adr.html',config={'displayModeBar':True}, output_type = 'div')


            output9=str("7. The highest Month over Month decrease in ADR occured from {} to {}, where the value fell by {}. The property needs to check on why there is a sudden dip.".format(d,a,b))
            output10=str("8. The highest Month over Month increase in ADR occured from {} to {}, where the value grew by {}. The property has the highest increase in daily revenue here.".format(h,e,f))
            
            fin1 = output1+os.linesep
            fin2 = os.linesep+output2+os.linesep+os.linesep+os.linesep+output3+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8+os.linesep+os.linesep+output9+os.linesep+os.linesep+output10
            img_text = str("The trend chart for the performance indicator - ADR is open and available in your browser for viewing") 
            
            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img)        
        
    
    @asyncio.coroutine
    async def revpar_func(self,hotel,month,turn_context:TurnContext): # Revenue per available room (RevPar) = Average daily rate(ADR) x occupancy rate 
        
        hotel_list = data_Hotel_fin.Hotel.unique().tolist()
        
        for i in range(len(hotel_list)):
            hotel_list[i] = hotel_list[i].lower()
            hotel_list[i] = hotel_list[i].replace(' ','')  
            
        if hotel.lower().replace(' ','') not in hotel_list:
            
            print('The available list of hotels are Hotel 1, Hotel 2, Hotel 3, Hotel 4. Please enter again')
            #rev_par()
            
        else:
            # Occupancy Rate
            data_Hotel_fin['Occupancy_Rate']=round((data_Hotel_fin['Room Sold']/data_Hotel_fin['Total Rooms']),2)
            data_occ = data_Hotel_fin[['Hotel','Month','Occupancy_Rate']]
            
            # ADR
            df = pd.merge(data_Finance[data_Finance['Sub Category']=='Room Revenue'][['Hotel','Month','CY']],data_Hotel_1[['Hotel','Month','Room Sold']], on = ['Hotel','Month'],how = 'left')
            subdf = df.groupby(['Hotel','Month']).agg({'CY':'sum','Room Sold':'mean'}).reset_index()
            subdf['adr'] = (-1)*round(subdf['CY']/subdf['Room Sold'],2)
            data_adr=subdf[['Hotel','Month','adr']]
            
            #final data frame
            
            data_fin = pd.merge(data_occ,data_adr,on=['Hotel','Month'])
            data_fin['revpar']=round(data_fin['Occupancy_Rate']*data_fin['adr'],2)
            
            ### REVPAR
            data_fin_v1 = data_fin[data_fin['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            data_fin_v1 = data_fin_v1[data_fin_v1.Month==month]
            a=data_fin_v1.revpar.sum()
            a_view=f"{a:,}"
            
            output1= str("The Revenue per available room for {} in {} is {} $".format(hotel,month,a_view))
            
            output2=str("I have collated some interesting insights about the property {} -".format(hotel))
            
            # Insights
            
            # Overall insights
            
            subdf = data_fin.groupby('Hotel').agg({'revpar':'mean'}).reset_index()
            subdf['revpar']=round(subdf['revpar'],2)
            data_Hotel_fin_w1=subdf[['Hotel','revpar']]
            data_Hotel_fin_w2=data_Hotel_fin_w1[data_Hotel_fin_w1.revpar == data_Hotel_fin_w1.revpar.max()]
            data_Hotel_fin_w3=data_Hotel_fin_w1[data_Hotel_fin_w1.Hotel.str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]

            q=data_Hotel_fin_w2.Hotel.sum()
            j=data_Hotel_fin_w2.revpar.sum()
            j_view = f"{j:,}"
            i=str(j_view)+' $'
            u=data_Hotel_fin_w3.Hotel.sum()
            m=data_Hotel_fin_w3.revpar.sum()
            m_view = f"{m:,}"
            k=round(j-m,2)
            k_view = f"{k:,}"
            k=str(k_view)+' $'

            if q==u:
                output3=str("1. The property {} you are interested in is the best performing hotel, with an revpar of {}".format(q,i))
            else:
                output3=str("1. The best performing property is: {}. {} has {} less revpar compared to the best performing hotel".format(q,hotel,k))

            # Hotel wise insights
        
            # Month with highest ADR
            data_fin_v1 = data_fin[data_fin['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            subdf = data_fin_v1.groupby('Month').agg({'revpar':'mean'}).reset_index()
            subdf['revpar']=round(subdf['revpar'],2)
            data_Hotel_fin_w1=subdf[['Month','revpar']]
            data_Finance_v4=data_Hotel_fin_w1[data_Hotel_fin_w1.revpar==data_Hotel_fin_w1.revpar.max()]
            y=data_Finance_v4.revpar.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output4 = str("2. The property had the best REVPAR of {} in month {}, making it the month where highest revenue is grossed in available rooms.".format(y,x))
            
            
            # Month with lowest ADR
            data_fin_v1 = data_fin[data_fin['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            subdf = data_fin_v1.groupby('Month').agg({'revpar':'mean'}).reset_index()
            subdf['revpar']=round(subdf['revpar'],2)
            data_Hotel_fin_w1=subdf[['Month','revpar']]
            data_Finance_v4=data_Hotel_fin_w1[data_Hotel_fin_w1.revpar==data_Hotel_fin_w1.revpar.min()]
            y=data_Finance_v4.revpar.sum()
            y=f"{y:,}"
            x=data_Finance_v4['Month'].sum()
            output5 = str("3. The property had the lowest REVPAR of {} in month {}, making it the month where lowest revenue is grossed in available rooms.".format(y,x))
            
            # Quarter with best performance
            data_Finance_v3 = data_Hotel_fin_w1.copy()
            cond = [data_Finance_v3.Month =='January',
            data_Finance_v3.Month =='February',
            data_Finance_v3.Month =='March',
            data_Finance_v3.Month =='April',
            data_Finance_v3.Month =='May',
            data_Finance_v3.Month =='June',
            data_Finance_v3.Month =='July',
            data_Finance_v3.Month =='August',
            data_Finance_v3.Month =='September',
            data_Finance_v3.Month =='October',
            data_Finance_v3.Month =='November',
            data_Finance_v3.Month =='December'

            ]
            val = ['Quarter 1','Quarter 1','Quarter 1','Quarter 2','Quarter 2','Quarter 2','Quarter 3','Quarter 3','Quarter 3','Quarter 4','Quarter 4','Quarter 4']

            data_Finance_v3['Quarter']=np.select(cond,val)
            subdf = data_Finance_v3.groupby('Quarter').agg({'revpar':'mean'}).reset_index()
            subdf['revpar'] = round(subdf['revpar'],2)
            data_Hotel_fin_w1=subdf[['Quarter','revpar']]
            data_Finance_vq1=data_Hotel_fin_w1[data_Hotel_fin_w1.revpar==data_Hotel_fin_w1.revpar.max()]
            x=data_Finance_vq1['Quarter'].sum()
            y=round(data_Finance_vq1['revpar'].sum(),0)
            y_view=f"{y:,}"
            output6 = str("4. The property had the best REVPAR in {} with {}.".format(x,y_view))
            
            
            
            # Month over Month
            data_fin_v1 = data_fin[data_fin['Hotel'].str.lower().str.replace(' ','')==hotel.lower().replace(' ','')]
            subdf = data_fin_v1.groupby('Month').agg({'revpar':'mean'}).reset_index()
            subdf['revpar']=round(subdf['revpar'],2)
            data_Hotel_fin_w1=subdf[['Month','revpar']]
            
            
            data_Finance_v3=Sort_Dataframeby_Month(data_Hotel_fin_w1,monthcolumnname='Month')
            data_Finance_v3=data_Finance_v3.reset_index()
            data_Finance_v3['index_1'] = data_Finance_v3['index']+1
            data_Finance_v4 = pd.merge(data_Finance_v3[['Month','revpar','index']],data_Finance_v3[['Month','revpar','index_1']],left_on='index',right_on='index_1',how='left')
            data_Finance_v4.rename({'Month_x':'Month','revpar_x':'CM','revpar_y':'LM'},axis=1,inplace=True)
            data_Finance_v4=data_Finance_v4[['Month','CM','LM','index']]
            data_Finance_v4['LM']=np.where(data_Finance_v4.LM.isnull(),data_Finance_v4.CM,data_Finance_v4.LM)
            data_Finance_v4['MOM'] = data_Finance_v4['CM']-data_Finance_v4['LM']
            data_Finance_vx=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.min()]
            data_Finance_vy=data_Finance_v4[data_Finance_v4.MOM==data_Finance_v4.MOM.max()]
            
            a=data_Finance_vx.Month.sum()
            b=round(abs(data_Finance_vx.MOM.sum()),2)
            b=f"{b:,}"
            c=data_Finance_vx['index'].sum()
            d=data_Finance_v4[data_Finance_v4['index']==(c-1)].Month.sum()

            e=data_Finance_vy.Month.sum()
            f=round(abs(data_Finance_vy.MOM.sum()),2)
            f=f"{f:,}"
            g=data_Finance_vy['index'].sum()
            h=data_Finance_v4[data_Finance_v4['index']==(g-1)].Month.sum()
            
            data = [go.Scatter(x=data_Finance_v3['Month'],y=data_Finance_v3['revpar'],marker_color='midnightblue',)]
            fig = go.Figure(data=data)
            fig.update_layout(title_text='RevPar Vs Month', title_x=0.5)
            img = plot(fig,filename='revpar.html',config={'displayModeBar':True}, output_type = 'div')


            output7=str("5. The highest Month over Month decrease in REVPAR occured from {} to {}, where the value fell by {}. The property needs to check on why there is a sudden dip.".format(d,a,b))
            output8=str("6. The highest Month over Month increase in REVPAR occured from {} to {}, where the value grew by {}. The property has the highest increase revenue per rooms available here.".format(h,e,f))
            
            fin1 = output1+os.linesep+os.linesep
            fin2 = os.linesep+output2+os.linesep+os.linesep+output3+os.linesep+os.linesep+output4+os.linesep+os.linesep+output5+os.linesep+os.linesep+output6+os.linesep+os.linesep+output7+os.linesep+os.linesep+output8
            img_text = str("The trend chart for the performance indicator - RevPar is open and available in your browser for viewing.")  
            
            
            await turn_context.send_activity(MessageFactory.text(fin1))
            await turn_context.send_activity(MessageFactory.text(fin2))
            await turn_context.send_activity(MessageFactory.text(img_text))
            displayHTML(img)
            
    # For any new bot response pattern, add the intent (pattern) to function mapping in the below 'mappings' dictionary
    # All the response functions are members of the MyBot class    
      
    @asyncio.coroutine
    async def getResponse(self,ints, intents_json,hotel,month,turn_context: TurnContext):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        
        mappings = {"occupancy rate":self.occupancy_rate_func, "gop":self.gop_func, "goppar":self.goppar_func, "alos":self.alos_func,"unknown":self.unknown_func,
        "adr": self.adr_func, "rooms_sold":self.rooms_sold_func,"ebitdap": self.ebitdaP_func,"revpar":self.revpar_func, "goodbye":self.goodbye_func, "restart":self.restart_func,
        "bussi_prom_expenses":self.bussi_prom_expenses_func, "fnb_revenue":self.fnb_revenue_func}
        
        for i in list_of_intents:
            for keys in mappings.keys():
                if tag == keys:
                    result = mappings[keys](hotel,month,turn_context)
                    return await result
        
            
                   
    # The bot will be initiated with the user inputs in the flow that they are defined (top to bottom fashion)
    # For any new user input, you have to make changes to the .py files in the data_models folder
            
    
    async def _fill_out_user_inputs(self, flow: ConversationFlow, inputs: UserInputs, turn_context: TurnContext):
        user_input = turn_context.activity.text.strip()

        # ask for name
        if flow.last_question_asked == Question.NONE:
            await turn_context.send_activity(
                MessageFactory.text("Hola! I am FredEY your Financial Assistant :) What is your name?")
               
            ) 
            flow.last_question_asked = Question.NAME

        # validate name then ask for age
        elif flow.last_question_asked == Question.NAME:
            validate_result = self._validate_name(user_input)
            if not validate_result.is_valid:
                await turn_context.send_activity(
                    MessageFactory.text(validate_result.message)
                )
            else:
                inputs.name = validate_result.value
                await turn_context.send_activity(
                    MessageFactory.text(f"Hey {inputs.name}...")
                )
                await turn_context.send_activity(
                    MessageFactory.text("What property are you interested in?")
                )
                flow.last_question_asked = Question.HOTEL

        # validate age then ask for date
        elif flow.last_question_asked == Question.HOTEL:
            validate_result = self._validate_hotel(user_input)
            if not validate_result.is_valid:
                await turn_context.send_activity(
                    MessageFactory.text(validate_result.message)
                )
            else:
                inputs.hotel = validate_result.value
                await turn_context.send_activity(
                    MessageFactory.text(f"I have the property of your interest as {inputs.hotel}.")
                )
                await turn_context.send_activity(
                    MessageFactory.text("What timeperiod are you looking at? (Month of the year)")
                )
                flow.last_question_asked = Question.MONTH

        # validate date and wrap it up
        elif flow.last_question_asked == Question.MONTH:
            validate_result = self._validate_date(user_input)
            inputs.month = validate_result.value
            if not validate_result.is_valid:
                await turn_context.send_activity(
                    MessageFactory.text(validate_result.message)
                )
            else:
                inputs.trigger = validate_result.value
                await turn_context.send_activity(
                    MessageFactory.text("Please type your query ")
                )
                flow.last_question_asked = Question.TRIGGER 
        
        elif flow.last_question_asked == Question.TRIGGER:
            validate_result = self._validate_date(user_input)
            if not validate_result.is_valid:
                await turn_context.send_activity(
                    MessageFactory.text(validate_result.message)
                )
            else:
                hotel = inputs.hotel
                month = inputs.month
                inputs.trigger = validate_result.value
                ints = self.predict_class(inputs.trigger, model)
                #bot_responded = self.getResponse(ints, intents,hotel,month, turn_context)
                await self.getResponse(ints, intents,hotel,month, turn_context)
                    
    

    async def on_message_activity(self, turn_context: TurnContext):
        inputs = await self.profile_accessor.get(turn_context, UserInputs)
        flow = await self.flow_accessor.get(turn_context, ConversationFlow)
        await self._fill_out_user_inputs(flow, inputs, turn_context)

    # Save changes to UserState and ConversationState
        await self.conversation_state.save_changes(turn_context)
        await self.user_state.save_changes(turn_context)
        
        

