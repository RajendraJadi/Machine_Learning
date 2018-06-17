
# coding: utf-8

# ### Data Description

# In[144]:


from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('InfosysHackathonProductbyZip.csv', sep=',',usecols=range(0, 9), encoding="ISO-8859-1", quotechar = "\'")
df.columns  = ['Job_Count','DeliveryDate','ToCity','ToState','ToZip','ToLatitude','ToLongitude','ProductDescription' ,'Client']
df.head()


# In[146]:


df.describe()
df.count()


# ## Date Preprocessing

# In[7]:


df.dropna()


# In[ ]:


np.any(df.isnull())


# In[126]:


df.corr()


# In[84]:


df.info()


# In[10]:



df['Job_Count'] = df.Job_Count.astype(int)


# In[12]:


df.info()


# ## Data Visualization

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
sns.distplot(df['ToLatitude'])


# In[14]:



sns.countplot(x='Job_Count',data=df[:1000])


# In[15]:


sns.boxplot(x='DeliveryDate',y='ToLongitude',data=df[:100])


# In[16]:


sns.lmplot(x='ToLatitude',y='ToLongitude',data=df[:1000])


# In[17]:


sns.lmplot(x='ToLatitude',y='ToLongitude',row='ToState',data=df[:100],palette='coolwarm')


# In[61]:


sns.jointplot(x='Job_Count',y='ToLatitude',data=df,kind='hex')


# In[20]:


exclude=['id','lat','long']
df.ix[:, df.columns.difference(exclude)].hist(alpha=0.7, figsize=(15, 15))
plt.show()


# In[22]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.distplot(df['Job_Count'])


# In[30]:


sns.stripplot(x="ToLongitude", y="ToLatitude",hue="Job_Count", data=df[:1000],jitter=True)


# In[143]:


sns.jointplot(x='ToLongitude',y='ToLatitude',data=df,color='red',kind='kde');


# In[31]:


sns.heatmap(df.corr())


# In[32]:


sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# In[33]:


sns.countplot(x='Client',data=df[:1000])


# In[34]:


exclude=['DeliveryDate','ToCity','ToState']
df.ix[:, df.columns.difference(exclude)].hist(alpha=0.7, figsize=(15, 5))
plt.show()


# In[62]:


sns.pairplot(df,hue='Job_Count',palette='coolwarm')


# # Linear Regression Model

# In[35]:


import numpy as np 
from abc import ABC, abstractmethod

# Super class for machine learning models 

class BaseModel(ABC):
    """ Super class for ITCS Machine Learning Class"""
    
    @abstractmethod
    def train(self, X, T):
        pass

    @abstractmethod
    def use(self, X):
        pass

    
class LinearModel(BaseModel):
    """
        Abstract class for a linear model 
        
        Attributes
        ==========
        w       ndarray
                weight vector/matrix
    """

    def __init__(self):
        """
            weight vector w is initialized as None
        """
        self.w = None

    def _check_matrix(self, mat, name):
        print(mat.shape)
        if len(mat.shape) != 2:
            raise ValueError(''.join(["Wrong matrix ", name]))
        
    # add a basis
    def add_ones(self, X):
        """
            add a column basis to X input matrix
        """
        self._check_matrix(X, 'X')
        return np.hstack((np.ones((X.shape[0], 1)), X))

    ####################################################
    #### abstract funcitons ############################
    @abstractmethod
    def train(self, X, T):
        """
            train linear model
            
            parameters
            -----------
            X     2d array
                  input data
            T     2d array
                  target labels
        """        
        pass
    
    @abstractmethod
    def use(self, X):
        """
            apply the learned model to input X
            
            parameters
            ----------
            X     2d array
                  input data
            
        """        
        pass 


# ### Least Squares

# In[36]:


# Linear Regression Class for least squares
class LinearRegress(LinearModel): 
    """ 
        LinearRegress class 
        
        attributes
        ===========
        w    nd.array  (column vector/matrix)
             weights
    """
    def __init__(self):
        LinearModel.__init__(self)
        
    # train lease-squares model
    def train(self, X, T):
        self._check_matrix(X,'X')
        self._check_matrix(T,'T')
        X1 = self.add_ones(X)
        self.w = np.linalg.lstsq(X1 , T)[0]
        
       
    
    # apply the learned model to data X
    def use(self, X):
              
        X1 = self.add_ones(X)
        
        return np.dot(X1 , self.w)
        #pass  ## TODO: replace this with your codes


# ### Least Mean Squares

# In[37]:


import collections # for checking iterable instance

# LMS class 
class LMS(LinearModel):
    """
        Lease Mean Squares. online learning algorithm
    
        attributes
        ==========
        w        nd.array
                 weight matrix
        alpha    float
                 learning rate
    """
    def __init__(self, alpha):
        LinearModel.__init__(self)
        self.alpha = alpha
    
    # batch training by using train_step function
    def train(self, X, T):
        for x, t in zip(X,T):
            self.train_step(x,t)
        
        
            
    # train LMS model one step 
    # here the x is 1d vector
    
    def train_step(self, x, t):
        if len(x.T.shape) != 2:
            x = np.insert(x, 0, 1).reshape(-1, 1)
            
        if self.w is None:
            self.w = np.zeros((x.shape[0], 1))
            
        self.w = self.w - (self.alpha * (self.w.T @ x - t) * x)

       

    
    # apply the current model to data X
    def use(self, X):
        X = self.add_ones(X)
        y = self.w.T @ X.T
        return y.T
        

           
        
        
        #pass  ## TODO: replace this with your codes


# In[38]:


# HERE follow are for my code tests.
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[39]:


X = np.linspace(0,10, 11).reshape((-1, 1))
T = -2 * X + 3.2

ls = LinearRegress()

ls.train(X, T)

plt.plot(ls.use(X))


# In[40]:


lms = LMS(0.1)
#X1=lms.add_ones(X)
for x, t in zip(X, T):
    lms.train_step(x, t)
    plt.plot(lms.use(X))


# In[41]:


lms.train(X, T)
plt.plot(lms.use(X))


# In[42]:


#X = np.linspace(0,10, 11).reshape((-1, 1))
#T = -2 * X + 3.2
X = df['ToLatitude'][:1000].values

#X = np.reshape(data1, (data1.shape[0], 1))
#print(X.shape)
#print(X)
#X = np.vstack([X, np.ones(len(X))]).T
X = np.reshape(X, (X.shape[0], 1))
print("x shape is :",X.shape)

T = df['ToLongitude'][:1000].values
T = np.reshape(T, (T.shape[0], 1))
#print(T.shape)
ls = LinearRegress()

w= ls.train(X, T)

plt.plot(X,ls.use(X))


# In[43]:


X = df['ToLatitude'][:1000]
X = X.values.reshape((-1, 1))
T = df['ToLongitude'][:1000]
T = T.values.reshape((-1, 1))
ls = LMS(0.001)

ls.train_step(X[0], T[0])
#plt.figure()
#plt.xlabel(fea)
#plt.ylabel(targetFea)
plt.plot(X,ls.use(X))


# In[47]:


X = df['ToLatitude'][:1000]
X = X.values.reshape((-1, 1))
T = df['Job_Count'][:1000]
T = T.values.reshape((-1, 1))
ls = LMS(0.001)

ls.train(X, T)
plt.figure()
#plt.xlabel(fea)
#plt.ylabel(targetFea)
plt.plot(X,ls.use(X))


# # Logistic Regression

# In[131]:


from sklearn.model_selection import train_test_split


# In[132]:


X = df[['Job_Count','DeliveryDate','ToCity','ToState','ToZip','ToLatitude','ToLongitude','ProductDescription' ,'Client']]
y = df['Job_Count']


# Split the data into training set and testing set using train_test_split

# In[133]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[134]:


from sklearn.linear_model import LogisticRegression


# In[135]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ### Predictions and Evaluations
# predicting values for the testing data.

# In[136]:


predictions = logmodel.predict(X_test)


# In[137]:


from sklearn.metrics import classification_report


# In[138]:


print(classification_report(y_test,predictions))

