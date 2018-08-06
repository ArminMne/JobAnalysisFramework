import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from rpy2.robjects import r,numpy2ri
from scipy.stats import gaussian_kde as kde
from pylab import plot,figure,subplot,fill,show,close,title,linspace
import re,pylab,pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
#style.use('bmh')
#style.use('seaborn-white')

#
# Code in onet.py needs to be run first to generate raw data (from within the data folders... db_11.. and db_18...
#

numpy2ri.activate()

#Database to generate pkl files from the O-net
onetdatabase = "db_20_0"
# Lookup table for abilities, activities and skills
cont=pd.read_csv(onetdatabase+'/Content Model Reference.txt','\t')
jd=pd.read_csv(onetdatabase+'/Occupation Data.txt','\t')
lookup=dict(cont[['Element ID','Element Name']].values)

FolderForWeights = "Final_Sigmoid_Weights"
#FolderForWeights = "New_Sigmoid_Weights2017"
FileForWeights = "sigmoid_weights_2015_GCC_oil&gas.pkl"
#FileForWeights = "sigmoid_weights_2015_GCC_banking&finance.pkl"
#FileForWeights = "sigmoid_weights_2015_USA_oil&gas.pkl"
#FileForWeights = "sigmoid_weights_2015_USA_banking&finance.pkl"

###########################
# Creating the data
############################
tmpDats=[]
infiles=[onetdatabase+"/Abilities.txt",onetdatabase+"/Work Activities.txt",onetdatabase+"/Skills.txt"]
for infile in infiles:
foo=pd.read_csv(infile,'\t')
foo=foo[foo['Scale ID']=='IM']
tmpDats.append(foo.pivot('O*NET-SOC Code','Element ID','Data Value'))

allDat=pd.concat(tmpDats,axis=1)
allDat.to_csv('RData.csv')

# we crated pickle files so this sector is under comment now #Armin
"""
#####################################
#Create Pickle files
#####################################
with open('alldat_2012_db_17_0.pkl', 'wb') as handle:
pickle.dump(allDat, handle)
"""


#All models explained here in details: http://scikit-learn.org/stable/modules/decomposition.html#fa #Armin
# Factor analysis version
#############################
# Let's get going!

filthresh=0.6
filtper=94
nComponents=15
numSkills=6 # Number of Skills to display
permitted=allDat.columns

for count1 in range(10):

ad2=allDat[permitted]
ad3=np.array(ad2) #added by Ioannis
# Factor Analysis
#lff=pd.DataFrame(np.array(r.factanal(ad2,nComponents)[1])).T
lff=pd.DataFrame(np.array(r.factanal(ad3,nComponents,scores='regression', rotation = "varimax")[1])).T
fit = r.factanal(ad3,nComponents, scores='regression', rotation = "varimax")
corr = fit[3]
scores= np.array(fit.rx2('scores'))

if True: # If want compatibility with ICIS paper set to False
filthresh=np.percentile(lff,filtper)

# Compiling list of "permitted" elements
permitted=[]
for count2 in range(len(lff[0])):
foo=ad2.columns[lff.iloc[count2,:].apply(abs)>filthresh]
if len(foo)>2:
permitted.extend(foo)
permitted=np.unique(permitted)
print "#Permitted: "+str(len(permitted))

#
columns=ad2.columns
#



# Producing the matrices of factors
factors=[]
fa_factorinds=[]
for c in range(15):
try:
nextvec=lff.iloc[c,:]
if sum(nextvec.apply(abs)>filthresh)>2:
fa_factorinds.append(c)
factvec=nextvec[nextvec.apply(abs)>filthresh].sort_values
(inplace=False,ascending=False)
factors.append(";".join([lookup[tmp] for tmp in [columns[tmp2] for tmp2 in factvec.index[0:numSkills]]]))
except IndexError:
break

ad_fa=columns
fa_factors=factors;

print("FAV_Factors: ",fa_factors)
"""
('FAV_Factors: ',
['Inductive Reasoning;Critical Thinking;Deductive Reasoning;Complex Problem Solving;Judgment and Decision Making;Active Learning',
'Troubleshooting;Equipment Maintenance;Repairing;Equipment Selection;Repairing and Maintaining Mechanical Equipment;Operation Monitoring',
'Peripheral Vision;Night Vision;Spatial Orientation;Glare Sensitivity;Sound Localization',
'Mathematics;Number Facility;Mathematical Reasoning',
'Coordinating the Work and Activities of Others;Developing and Building Teams;Guiding, Directing, and Motivating Subordinates'])
"""

#####################
# NMF Version
#####################

from sklearn.decomposition import NMF

nComponents=20
nSkills=6
permitted=allDat.columns
filtper=90 # Percentile to threshold on

for count1 in range(8):

ad2=allDat[permitted]

nmf=NMF(n_components=nComponents,random_state=10)
nmf.fit(ad2)
filthresh=np.percentile(nmf.components_,filtper)

lf=pd.DataFrame(nmf.components_)

permitted=[]
for count2 in range(nComponents):
foo=ad2.columns[lf.iloc[count2,:]>filthresh]
if len(foo)>2:
permitted.extend(foo)
permitted=np.unique(permitted)

print "#Permitted: "+str(len(permitted))

# Producing the matrices of factors
filthresh2=np.percentile(nmf.components_,98)
factors=[]
factorinds=[]
for c in range(nComponents):
try:
nextvec=lf.iloc[c,:]
if sum(nextvec>filthresh2)>2:
factorinds.append(c)
topthree=nextvec.sort_values(ascending=False,inplace=False).index[0:nSkills]
factors.append(";".join([lookup[tmp] for tmp in [ad2.columns[tmp2] for tmp2 in topthree]]))
except IndexError:
break
except ValueError:
pass

nmf_factors=factors;

print("NMF_Factors: ",nmf_factors)
"""
('NMF_Factors: ',
['Oral Comprehension;Active Listening;Oral Expression;Inductive Reasoning;Getting Information;Problem Sensitivity',
'Performing General Physical Activities;Stamina;Extent Flexibility;Trunk Strength;Gross Body Coordination;Static Strength',
'Repairing;Equipment Maintenance;Repairing and Maintaining Electronic Equipment;Repairing and Maintaining Mechanical Equipment;Troubleshooting;Controlling Machines and Processes',
'Mathematics;Mathematical Reasoning;Number Facility;Interacting With Computers;Processing Information;Analyzing Data or Information',
'Interacting With Computers;Quality Control Analysis;Programming;Troubleshooting;Operation Monitoring;Technology Design',
'Interacting With Computers;Performing Administrative Activities;Documenting/Recording Information;Processing Information;Communicating with Persons Outside Organization;Organizing, Planning, and Prioritizing Work'])
"""

######################################
# Creating tables and figures
######################################


################################
# distributions.png
# Run within the db_11 version
################################
numPlots=8
skills=[int(tmp) for tmp in linspace(1,125,numPlots)]
figure()
for count in range(numPlots):

ax=subplot(numPlots/2,2,count+1)

y=kde(allDat.iloc[:,skills[count]]).evaluate(linspace(0,6,100))
x=linspace(0,6,100);

ax.fill(x,y,'tan')
ax.set_xticks([])
ax.set_yticks([])

atext=lookup[allDat.columns[skills[count]]]
awords=re.compile('\s+').split(atext)
if awords>3:
atext=' '.join(awords[0:3])+"\n"+' '.join(awords[3:])
ax.set_ylim(top=ax.get_ylim()[1]*1.85)
yloc=0.6
else:
ax.set_ylim(top=ax.get_ylim()[1])
yloc=0.5

#ax.text(0.5,yloc,atext,transform=ax.transAxes)
ax.text(0.1,yloc,atext,transform=ax.transAxes)
show()

#######################
# loadings.png
#######################
factloadings=abs(lff)
factloadings=factloadings.T.apply(lambda x:x.sort_values(ascending=False,inplace=False).reset_index(drop=True)).T
#factloadings=factloadings.iloc[:,0:15]

factloadings=factloadings-factloadings.min().min()
normvec=factloadings.max(axis=1)
normmat=np.tile(normvec,(factloadings.shape[1],1)).T
#factloadings=1-factloadings/factloadings.max().max()
factloadings=1-factloadings/normmat
factloadings=factloadings.iloc[:,0:20]

nmfloadings=lf
nmfloadings=nmfloadings.T.apply(lambda x:x.sort_values(ascending=False,inplace=False).reset_index(drop=True)).T
#nmfloadings=nmfloadings.iloc[:,0:15]

nmfloadings=nmfloadings-nmfloadings.min().min()
normvec=nmfloadings.max(axis=1)
normmat=np.tile(normvec,(nmfloadings.shape[1],1)).T
#nmfloadings=1-nmfloadings/nmfloadings.max().max()
nmfloadings=1-nmfloadings/normmat
nmfloadings=nmfloadings.iloc[:,0:20]

ax1=subplot(1,2,1)
ax2=subplot(1,2,2)
ax1.matshow(factloadings.iloc[:,0:20],cmap=pylab.cm.gray,
interpolation="nearest")
ax1.set_title("Loadings (Factor Analysis)")
ax1.spines['left'].set_position(('outward', 10))
ax1.spines['bottom'].set_position(('outward', 10))
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

ax2.matshow(nmfloadings.iloc[0:15,0:20],cmap=pylab.cm.gray,
interpolation="nearest")
ax2.set_title("Loadings (NMF)")
ax2.spines['left'].set_position(('outward', 10))
ax2.spines['bottom'].set_position(('outward', 10))
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
show()

######################
# Factor skills table
######################
pd.set_option('display.max_colwidth', -1)
def printFactorTable(labels,infacts,factsep=";"):
factors=[", ".join(re.split(factsep,tmp)[0:3]) for tmp in infacts]
factDF=pd.DataFrame(zip(labels,factors),columns=["Factor","Skills"])
return factDF.to_latex(index=False)
"""
#db_11
favFactorLabels=['Leadership','Manual','Equipment','Vehicle Operation','Perception','Mathematical']

#db_13
favFactorLabels=['Manual', 'Leadership', 'Equipment', 'Social', 'Research', 'Perception',  'Vehicle Operation','Creativity','Mathematical']
#db_13, perc=94%
favFactorLabels=['Manual', 'Vehicle Operation',  'Equipment', 'Leadership', 'Perception', 'Interpersonal', 'Creativity']
#db_13, filthre =0.65, no per
favFactorLabels=['Manual',  'Equipment', 'Leadership', 'Vehicle Operation',  'Perception', 'Interpersonal']

#db_15,16
favFactorLabels=['Cognitive','Equipment', 'Manual','Vehicle Operation','Leadership','Mathematical']
#db_15,perc=92.5
favFactorLabels=['Equipment', 'Cognitive', 'Leadership', 'Manual','Vehicle Operation','Perception','Mathematical', 'Creativity'] #fail
#db_15,perc=93.0
favFactorLabels=['Equipment', 'Cognitive', 'Leadership', 'Manual','Vehicle Operation','Mathematical', 'Perception', 'Creativity'] #fail
#db_15,perc=94.0
favFactorLabels=['Cognitive', 'Manual', 'Equipment', 'Vehicle Operation','Leadership', 'Mathematical'] #average
#db_15,filthre =0.65, no per
favFactorLabels=['Equipment','Cognitive', 'Vehicle Operation','Leadership/Supervision','Manual', 'Mathematical'] #average

#db_17
favFactorLabels=['Cognitive','Equipment', 'Vehicle Operation', 'Manual','Leadership', 'Intepersonal', 'Mathematical']
#db_17, perc=93%
favFactorLabels=['Cognitive',  'Equipment', 'Leadership', 'Manual', 'Vehicle Operation', 'Intepersonal', 'Mathematical']
#db_17, perc=94%
favFactorLabels=['Cognitive', 'Manual', 'Equipment',  'Vehicle Operation', 'Leadership', 'Mathematical']
#db_17, filthre =0.65, no per
favFactorLabels=['Equipment', 'Cognitive', 'Vehicle Operation', 'Manual', 'Leadership']
"""
#db_20
favFactorLabels=['Problem Solving','Equipment','Perception','Mathematical','Supervision']
"""
('FAV_Factors: ',
Problem solving - 'Inductive Reasoning;Critical Thinking;Deductive Reasoning;Complex Problem Solving;Judgment and Decision Making;Active Learning',
Equipment - 'Troubleshooting;Equipment Maintenance;Repairing;Equipment Selection;Repairing and Maintaining Mechanical Equipment;Operation Monitoring',
Perception - 'Peripheral Vision;Night Vision;Spatial Orientation;Glare Sensitivity;Sound Localization',
Mathematical - 'Mathematics;Number Facility;Mathematical Reasoning',
Supervision - 'Coordinating the Work and Activities of Others;Developing and Building Teams;Guiding, Directing, and Motivating Subordinates'])
"""

#db_20
nmfFactorLabels=['Cognitive','Manual','Equipment','Mathematics','Quality Control Analysis','Management']
"""
('NMF_Factors: ',
Cognitive - 'Oral Comprehension;Active Listening;Oral Expression;Inductive Reasoning;Getting Information;Problem Sensitivity',
Manual - 'Performing General Physical Activities;Stamina;Extent Flexibility;Trunk Strength;Gross Body Coordination;Static Strength',
Equipment - 'Repairing;Equipment Maintenance;Repairing and Maintaining Electronic Equipment;Repairing and Maintaining Mechanical Equipment;Troubleshooting;Controlling Machines and Processes',
Mathematics - 'Mathematics;Mathematical Reasoning;Number Facility;Interacting With Computers;Processing Information;Analyzing Data or Information',
Quality Control Analysis (Conducting tests and inspections of products, services, or processes to evaluate quality or performance. ) - 'Interacting With Computers;Quality Control Analysis;Programming;Troubleshooting;Operation Monitoring;Technology Design',
Management - 'Interacting With Computers;Performing Administrative Activities;Documenting/Recording Information;Processing Information;Communicating with Persons Outside Organization;Organizing, Planning, and Prioritizing Work'])
"""

#Find all skills on: http://www.onetonline.org/skills/

LaTeX=False
if LaTeX:
print printFactorTable(favFactorLabels,fa_factors)
print printFactorTable(nmfFactorLabels,nmf_factors)

# code bellow print automatically top 3 skills under shorten name for example:
# {Cognitive}: Inductive Reasoning; Critical Thinking; Deductive Reasoning
print "\\begin{enumerate}"
for count in range(len(favFactorLabels)): # FA-V
print "\item \\textbf{"+favFactorLabels[count]+"}: "+"; ".join(fa_factors[count].split(';')[0:3])
print "\\end{enumerate}"

print "\\begin{enumerate}"
for count in range(len(nmfFactorLabels)): # NMF
print "\item \\textbf{"+nmfFactorLabels[count]+"}: "+"; ".join(nmf_factors[count].split(';')[0:3])
print "\\end{enumerate}"

###################################
# Linear regression 2008-2015 NMF
###################################
allDat2008=(pickle.load(open("alldat_2008_db_13_0.pkl"))[ad2.columns]).dropna()
allDat2015=pickle.load(open("alldat_2015_db_20_0.pkl"))[ad2.columns]
allDat2015=(allDat2015.loc[allDat2008.index]).dropna()
allDat2008=allDat2008.loc[allDat2015.index]

# Adding Weights NMF armin
weights2008=pd.read_pickle(FolderForWeights+"/"+FileForWeights)
weights2008=weights2008.set_index([0])
weights2008=weights2008.loc[allDat2008.index]

ser = pd.Series(weights2008[1])
func = lambda x: np.asarray(x) * np.asarray(ser)
allDat2008weights=allDat2008.apply(func)
allDat2015weights=allDat2015.apply(func)
#we can do same as above with code result = dataframe.mul(series, axis=0)
#

# Projecting onto NMF loadings
nmfacts2008=nmf.transform(allDat2008)[:,factorinds]
nmfacts2015=nmf.transform(allDat2015)[:,factorinds]

# Normalization
norm_nmfacts2008=nmfacts2008-np.tile(np.mean(nmfacts2008,axis=0),
(len(nmfacts2008),1))

norm_nmfacts2015=nmfacts2015-np.tile(np.mean(nmfacts2015,axis=0),
(len(nmfacts2015),1))

# print nmfacts2008.shape
# print len(nmfacts2008)
# Adding column of ones to find the intercept
nmfacts2008=np.concatenate((nmfacts2008,np.ones((len(nmfacts2008),1))),axis=1) #was before 753,1
#norm_nmfacts2008=np.concatenate((norm_nmfacts2008,np.ones((753,1))),axis=1)

# Setting up the linear regression models
linregs=[LinearRegression(copy_X=True,fit_intercept=False) for tmp in range(len(nmfFactorLabels))]
rsq=[]
for count in range(len(nmfFactorLabels)):
#linregs[count].fit(norm_nmfacts2008,norm_nmfacts2015[:,count])
#rsq.append(linregs[count].score(norm_nmfacts2008,norm_nmfacts2015[:,count]))
linregs[count].fit(nmfacts2008,nmfacts2015[:,count])
rsq.append(linregs[count].score(nmfacts2008,nmfacts2015[:,count]))

# Generating table
coeftableNMF=[]
for count in range(len(nmfFactorLabels)):
coeftableNMF.append(list(linregs[count].coef_)+[rsq[count]])
cdfNMF=pd.DataFrame(np.array(coeftableNMF).T,columns=
nmfFactorLabels,index=nmfFactorLabels+['(Intercept)','Rsq'])
print "cdfNMF: ", cdfNMF.applymap(lambda x:int(x*10000)/10000.0).to_latex()


#############################################
#same with weights###########################
#############################################
# Projecting onto NMF loadings
nmfacts2008weights=nmf.transform(allDat2008weights)[:,factorinds]
nmfacts2015weights=nmf.transform(allDat2015weights)[:,factorinds]

# Normalization
norm_nmfacts2008weights=nmfacts2008weights-np.tile(np.mean
(nmfacts2008weights,axis=0),(len(nmfacts2008weights),1))
norm_nmfacts2015weights=nmfacts2015weights-np.tile(np.mean
(nmfacts2015weights,axis=0),(len(nmfacts2015weights),1))

# Adding column of ones to find the intercept
nmfacts2008weights=np.concatenate((nmfacts2008weights,
np.ones((len(nmfacts2008weights),1))),axis=1)

# Setting up the linear regression models
linregs=[LinearRegression(copy_X=True,fit_intercept=False) for tmp in range(len(nmfFactorLabels))]
rsq=[]
for count in range(len(nmfFactorLabels)):
#linregs[count].fit(norm_nmfacts2008weights,norm_nmfacts2015weights[:,count])

linregs[count].fit(nmfacts2008weights,nmfacts2015weights[:,count])
rsq.append(linregs[count].score(nmfacts2008weights,nmfacts2015weights[:,count]))

# Generating table
coeftableNMF=[]
for count in range(len(nmfFactorLabels)):
coeftableNMF.append(list(linregs[count].coef_)+[rsq[count]])
cdfNMFweights=pd.DataFrame(np.array(coeftableNMF).T,columns=
nmfFactorLabels,index=nmfFactorLabels+['(Intercept)','Rsq'])
print "cdfNMFweights: ", cdfNMFweights.applymap(lambda x:int(x*10000)/10000.0).to_latex()

####################################
# Linear regression 2008-2015 FA-V
####################################
allDat2008FAV=(pickle.load(open("alldat_2008_db_13_0.pkl"))[ad_fa]).dropna()
allDat2015FAV=pickle.load(open("alldat_2015_db_20_0.pkl"))[ad_fa]
allDat2015FAV=(allDat2015FAV.loc[allDat2008FAV.index]).dropna()
allDat2008FAV=allDat2008FAV.loc[allDat2015FAV.index]

# Adding Weights FA-V armin
weights2008=pd.read_pickle(FolderForWeights+"/"+FileForWeights)
weights2008=weights2008.set_index([0])
weights2008=weights2008.loc[allDat2008FAV.index]

ser = pd.Series(weights2008[1])
func = lambda x: np.asarray(x) * np.asarray(ser)
allDat2008FAVweights=allDat2008FAV.apply(func)
allDat2015FAVweights=allDat2015FAV.apply(func)
#

# Projecting into FA-V loadings

norm_allDat2008FAV=np.array(allDat2008FAV)-np.tile(np.mean
(np.array(allDat2008FAV),axis=0),(len(allDat2008FAV),1))

norm_allDat2015FAV=np.array(allDat2015FAV)-np.tile(np.mean
(np.array(allDat2008FAV),axis=0),(len(allDat2008FAV),1))


favfacts2008=np.dot(np.dot(norm_allDat2008FAV,np.array(r.solve(np.array(corr)))), lff.T)[:,fa_factorinds]
favfacts2015=np.dot(np.dot(norm_allDat2015FAV,np.array(r.solve(np.array(corr)))), lff.T)[:,fa_factorinds]


# Adding column of ones to find the intercept
#favfacts2008=np.concatenate((favfacts2008,np.ones((753,1))),axis=1)
favfacts2008=np.concatenate((favfacts2008,np.ones((len(favfacts2008),1))),axis=1)

# Setting up the linear regression models
linregs=[LinearRegression(copy_X=True,fit_intercept=False, normalize=True) for tmp in range(len(favFactorLabels))]
rsq=[]
for count in range(len(favFactorLabels)):
#linregs[count].fit(norm_nmfacts2008,norm_nmfacts2015[:,count])
#rsq.append(linregs[count].score(norm_nmfacts2008,norm_nmfacts2015[:,count]))
linregs[count].fit(favfacts2008,favfacts2015[:,count])
rsq.append(linregs[count].score(favfacts2008,favfacts2015[:,count]))

# Generating table
coeftableFAV=[]
for count in range(len(favFactorLabels)):
coeftableFAV.append(list(linregs[count].coef_)+[rsq[count]])
cdfFAV=pd.DataFrame(np.array(coeftableFAV).T,columns=
favFactorLabels,index=favFactorLabels+['(Intercept)','Rsq'])
print "cdfFAV: ", cdfFAV.applymap(lambda x:int(x*10000)/10000.0).to_latex()


#############################################
#same with weights###########################
#############################################
# Projecting into FA-V loadings

norm_allDat2008FAVweights=np.array(allDat2008FAVweights)-np.tile
(np.mean(np.array(allDat2008FAVweights),axis=0),
(len(allDat2008FAVweights),1))

norm_allDat2015FAVweights=np.array(allDat2015FAVweights)-np.tile
(np.mean(np.array(allDat2008FAVweights),axis=0),
(len(allDat2008FAVweights),1))

favfacts2008weights=np.dot(np.dot(norm_allDat2008FAVweights,
np.array(r.solve(np.array(corr)))), lff.T)[:,fa_factorinds]
favfacts2015weights=np.dot(np.dot(norm_allDat2015FAVweights,
np.array(r.solve(np.array(corr)))), lff.T)[:,fa_factorinds]


# Adding column of ones to find the intercept

favfacts2008weights=np.concatenate((favfacts2008weights,np.ones
((len(favfacts2008weights),1))),axis=1)

# Setting up the linear regression models
linregs=[LinearRegression(copy_X=True,fit_intercept=False, normalize=True) for tmp in range(len(favFactorLabels))]
rsq=[]
for count in range(len(favFactorLabels)):
#linregs[count].fit(norm_nmfacts2008,norm_nmfacts2015[:,count])
#rsq.append(linregs[count].score(norm_nmfacts2008,norm_nmfacts2015[:,count]))
linregs[count].fit(favfacts2008weights,favfacts2015weights[:,count])
rsq.append(linregs[count].score(favfacts2008weights,favfacts2015weights[:,count]))

# Generating table
coeftableFAV=[]
for count in range(len(favFactorLabels)):
coeftableFAV.append(list(linregs[count].coef_)+[rsq[count]])
cdfFAVweights=pd.DataFrame(np.array(coeftableFAV).T,columns=
favFactorLabels,index=favFactorLabels+['(Intercept)','Rsq'])
print "cdfFAVweights: ", cdfFAVweights.applymap(lambda x:int(x*10000)/10000.0).to_latex()


###################
# Visualize occupations with 5 top/bottom weights
top5_weights=pd.DataFrame(weights2008.sort_values([1], ascending=False)[:5])
top5_weights=pd.DataFrame(pd.merge(top5_weights, jd, left_index=True, right_on='O*NET-SOC Code', how='inner'),columns=[1,'Title'])

# Plot histogram using matplotlib bar()
indexes = np.arange(len(top5_weights))
width = 0.5
plt.bar(indexes, np.array(top5_weights[1]), width, color='green')
#plt.xticks(indexes + width * 0.5, np.array(top5_weights['Title']))
plt.xticks(indexes + width * 0.5, np.array(top5_weights['Title']))
plt.title("Occupations with 5 top weights")
plt.xlabel("Occupations")
plt.ylabel("Weights")
plt.show()

bottom5_weights=pd.DataFrame(weights2008.sort_values([1], ascending=True))
bottom5_weights=pd.DataFrame(pd.merge(bottom5_weights, jd, left_index=True, right_on='O*NET-SOC Code', how='inner'),columns=[1,'Title'])
#print ("Bottom 5 weights: ",bottom5_weights.head())


# Plot Factors Shifts between 2008  in 2015 NMF FACTORS
plt.rcParams.update({'font.size': 16})
labels=nmfFactorLabels
fig, ax1 = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(left=0.115, right=0.88)
pos = np.arange(len(labels)) + 0.5    # Center bars on the Y-axis ticks
rects = ax1.barh(pos, np.array(cdfNMF.loc['(Intercept)'])*(-1), align='center', height=0.5, color='blue', label='Without weights')
rectsweights = ax1.barh(pos, np.array(cdfNMFweights.loc['(Intercept)'])*(-1), align='center', height=0.25, color='red', label='With weights')

ax1.axis([-0.12, 0.12, 0, 6])

pylab.yticks(pos, labels)
ax1.xaxis.grid(True)
# set the tick locations
#ax2 = plt.subplots(figsize=(9, 7))
ax2.set_yticks(pos)
# set the tick labels
# make sure that the limits are set equally on both yaxis so the ticks line up
ax2.set_ylim(ax1.get_ylim())

plt.title("NMF Factors Shifts between 2008 and 2015")
plt.xlabel("Change (in Standard Deviations)")
#Plot a solid vertical gridline to highlight the median position
plt.plot([0, 0], [0, 9], 'black', alpha=0.75)
#plt.ylabel("Factors")
plt.legend()
plt.show()


# Plot Factors Shifts between 2008 in 2015 FAV FACTORS
labels=favFactorLabels
fig, ax1 = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(left=0.115, right=0.88)
pos = np.arange(len(labels)) + 0.5    # Center bars on the Y-axis ticks
rects = ax1.barh(pos, np.array(cdfFAV.loc['(Intercept)'])*(-1), align='center', height=0.5, color='blue', label='Without weights')
rectsweights = ax1.barh(pos, np.array(cdfFAVweights.loc['(Intercept)'])*(-1), align='center', height=0.25, color='red', label='With weights')

ax1.axis([-1.8, 1.8, 0, 5])

pylab.yticks(pos, labels)
ax1.xaxis.grid(True)
# set the tick locations
#ax2 = plt.subplots(figsize=(9, 7))
ax2.set_yticks(pos)
# set the tick labels
# make sure that the limits are set equally on both yaxis so the ticks line up
ax2.set_ylim(ax1.get_ylim())

plt.title("FA Factors Shifts between 2008 and 2015")
plt.xlabel("Change (in Standard Deviations)")
#Plot a solid vertical gridline to highlight the median position
plt.plot([0, 0], [0, 9], 'black', alpha=0.75)
#plt.ylabel("Factors")
plt.legend()
plt.show()
