#import the library used to query a website
import urllib2
#import the Beautiful soup functions to parse the data returned from the website
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
'Accept-Encoding': 'none',
'Accept-Language': 'en-US,en;q=0.8',
'Connection': 'keep-alive'}

#first step to generate links
alllinks=[]
alllinksJOBS=""
finallinks = []

numberofpages = 20 #THIS IS IMPORTANT HOW MANY PAGES TO DOWNLOAD
for x in range(numberofpages):
#alllinks.append("https://www.naukrigulf.com/healthcare-jobs-in-uae-%d" %x)
alllinks.append("https://www.naukrigulf.com/oil-and-gas-jobs-%d?industry=36" % x)

#iterate over all links
for links in alllinks:
req = urllib2.Request(links, headers=hdr)
page = urllib2.urlopen(req)
naukrigulfHtml = page.read()
page.close()

soup = BeautifulSoup(naukrigulfHtml, "html.parser")  # Parse the html in the 'page' variable, and store it in Beautiful Soup format
NaukriAllLinks = soup.find_all("a")  # we find only the links which start by "a" HTML code

for links in NaukriAllLinks:
tmplink = links.get('href')  # inside all links with "a" we want only those with "href" HTML code
alllinksJOBS = alllinksJOBS + "\n" + str(tmplink)  # make one big string will all links

text_file = open("alllinksNAUKRI.txt", "w")  # store it in txt file for further processing
text_file.write(alllinksJOBS)
text_file.close()

openfile = open("alllinksNAUKRI.txt")
for line in openfile:
# if re.search(r"^/pagead/clk.*", line, re.MULTILINE): #using regular expression we take only the valid links for jobs
#     finallinks.append(line.rstrip()) #store them in list for further processing (to remove duplicate links)
if re.search(r"^https://www.naukrigulf.com/.*\d{12}$", line, flags=re.MULTILINE):  # using regular expression we take only the valid links for jobs
finallinks.append(line.rstrip()) #store them in list for further processing (to remove duplicate links)
openfile.close()

#here are the JOB links from all INDEED pages
FinalJobLinks = np.unique(finallinks) #we take only unique job links

print len(FinalJobLinks), "Jobs to download:"

#second part to save the data from the links above
df = pd.DataFrame(columns=['Title','Description']) #create empty data frame
try:
for x in range(len(FinalJobLinks)):
print "We're on job number %d" %(x)

NAUKRIHealthCare = FinalJobLinks[x] #specify the job url
req2 = urllib2.Request(NAUKRIHealthCare, headers=hdr)
page = urllib2.urlopen(req2) #Query the website and return the html to the variable 'page'
soup = BeautifulSoup(page, "html.parser") #Parse the html in the 'page' variable, and store it in Beautiful Soup format

print NAUKRIHealthCare
try:
title = soup.h1.text  # get title
except AttributeError:
print "'NoneType' object has no attribute text"

tmplist = re.split(r"Job Type", soup.select_one("p[class*=cl]").text)
description = tmplist[0].replace("Job Summary", '').replace("Job Description", '') #get description

title = title.strip() #remove blank characters
title = title.encode('utf-8') #put utf8 encoding because of arabic and some other special characters
description = description.strip()
description = description.encode('utf-8')


df = df.append({
"Title": title,
"Description":  description
}, ignore_index=True)
print df.loc[x]
except urllib2.HTTPError:
print "HTTP Error 500 happened"

df.to_csv('NaukriGulfOilGas5.csv')
