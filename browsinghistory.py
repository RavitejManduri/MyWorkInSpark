##import browserhistory as bh
##dict_obj = bh.get_browserhistory()
##dict_obj.keys()
##dict_obj['chrome'][0]/*
def parse(url):
	try:
		parsed_url_components = url.split('//')
		sublevel_split = parsed_url_components[1].split('/', 1)
		domain = sublevel_split[0].replace("www.", "")
		return domain
	except IndexError:
		print ("URL format error!")
import os
import sqlite3
data_path=r"C:\Users\ravit\AppData\Local\Google\Chrome\User Data\Default"
files = os.listdir(data_path)
history_db = os.path.join(data_path, 'history')
c = sqlite3.connect(history_db)
cursor = c.cursor()

select_statement = "SELECT urls.url, urls.visit_count,datetime(visit_time / 1000000 + (strftime('%s', '1601-01-01')), 'unixepoch') FROM urls, visits WHERE urls.id = visits.url and datetime(visit_time / 1000000 + (strftime('%s', '1601-01-01')), 'unixepoch') >2020-03-28;"

cursor.execute(select_statement)

results = cursor.fetchall()
countlist={}
for url, count,date in results:
	result=parse(url)
	if (date>"2020-03-14" and date<"2020-03-21"):
		if result not in countlist.keys():
			countlist[result] = 1
		else:
			x = countlist[result]
			countlist[result] = x + 1
print(countlist)


