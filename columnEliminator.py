#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:28:44 2018

@author: emre
"""
import csv

with open("NYPD_Motor_Vehicle_Collisions.csv") as csv_file:
    rdr= csv.reader( csv_file, delimiter=',' )
    with open("NYPD_Motor_Vehicle_Collisions_clean.csv","wb") as f:
        wtr= csv.writer(f)
        for r in rdr:
            del r[6]
            wtr.writerow(r)
            
