# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

link_path = "F:/kdd/dataSets/training/links (table 3).csv"
routes_path = "F:/kdd/dataSets/training/routes (table 4).csv"
links = pd.read_csv(link_path,encoding='utf-8')
routes = pd.read_csv(routes_path,encoding='utf-8')
    
def getLinkinfo():
    in_nums = np.array(links['in_top'])
    out_nums = np.array(links['out_top'])
    link_ids = np.array(links['link_id'])
    linklengths = np.array(links['length'])
    linklanes = np.array(links['lanes'])
    length = len(link_ids)
    linknum = {}
    linklength = {}
    linklane = {}
    for i in range(length):
        innum = 0
        if not in_nums[i]=="":
            innum = len(str(in_nums[i]).split(","))
        outnum = 0
        if not out_nums[i]=="":
            outnum = len(str(out_nums[i]).split(","))
        link_id = str(link_ids[i])
        linknum[link_id] = innum+outnum
        linklength[link_id] = linklengths[i]
        linklane[link_id] = linklanes[i]
    return linknum,linklength,linklane
    
def getLinkout():
    in_nums = np.array(links['in_top'])
    out_nums = np.array(links['out_top'])
    link_ids = np.array(links['link_id'])
    linkin = {}
    linkout = {}
    for i in range(len(link_ids)):
        linkin[link_ids[i]]=[]
        linkout[link_ids[i]] = []
        linkin[link_ids[i]] = str(in_nums[i]).split(",")
        linkout[link_ids[i]] = str(out_nums[i]).split(",")
    return linkin,linkout
    
def getRouteslengthandWidth():
    intersection_ids =  np.array(routes['intersection_id'])
    tollgate_ids = np.array(routes['tollgate_id'])
    link_seqs = np.array(routes['link_seq'])
    resdic = {}
    length = len(intersection_ids)
    linknum,linklength,linklane = getLinkinfo()
    pathTotallength = {}
    pathTotalWidth = {}
    print(linklength)
    for i in range(length):
        link_seq = link_seqs[i]
        link_arr = link_seq.split(",")
        l = 0
        w = 0
        for link in link_arr:
            l += linklength[link]
        for link in link_arr:
            w += linklane[link]*1.0/linknum[link]*linklength[link]/l
        routes_id = str(intersection_ids[i])+'-'+str(tollgate_ids[i]);
        pathTotalWidth[routes_id] = str(w)
        pathTotallength[routes_id] = str(l)
    return pathTotallength,pathTotalWidth
 
def getRoutesDic():
    intersection_ids =  np.array(routes['intersection_id'])
    tollgate_ids = np.array(routes['tollgate_id'])
    link_seqs = np.array(routes['link_seq'])
    resdic = {}
    length = len(tollgate_ids)
    for i in range(length):
        id = str(intersection_ids[i])+'-'+str(tollgate_ids[i])
        resdic[id] = link_seqs[i].split(',')
    print(resdic)
    return resdic
    
def writetoFile(filepath, pathTotallength, pathTotalWidth):
    fw = open(filepath, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"total_length"', '"average_width"']) + '\n')
    
    for k,v in pathTotallength.items():
        out_line = ','.join(['"' + k.split('-')[0] + '"', '"' + k.split('-')[1] + '"',
                                 '"'+v+ '"',
                                 '"' + pathTotalWidth[k] + '"']) + '\n'
        fw.writelines(out_line)
    fw.close()
    
    
if __name__ =="__main__":
    pathTotallength,pathTotalWidth = getRouteslengthandWidth()
    writetoFile("F:/kdd/dataSets/training/widthandlength.csv",pathTotallength,pathTotalWidth)
            
            
            
            
        
    