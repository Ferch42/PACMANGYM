import matplotlib.pyplot as plt

# Your three lists
list1 = [1440.8, 2684.2, 1911.9, 2074.9, 1667.8, 2465.0, 1515.7, 2127.4, 1428.5, 1405.2, 1999.4, 1774.9, 1352.9, 2059.7, 1610.9, 1218.5, 1357.3, 1050.5, 1358.6, 1540.7, 1418.9, 1711.9, 1214.2, 2935.0, 1723.1, 1505.1, 1966.1, 1611.6, 1190.9, 834.9, 750.5, 1992.5, 1502.4, 1476.5, 1528.9, 1483.0, 1202.2, 1643.0, 1106.9, 1561.1, 1857.0, 958.3, 1466.3, 1784.0, 1212.1, 1829.0, 2191.5, 1645.5, 1282.3, 1139.1, 1738.9, 1823.2, 1793.1, 897.7, 1123.7, 1158.7, 1780.1, 1388.2, 1286.9, 1007.5, 1022.6, 1332.6, 1312.9, 1586.2, 1184.6, 1639.6, 1103.0, 1453.8, 1710.1, 1094.7, 930.9, 1025.1, 1403.9, 1432.4, 931.4, 731.8, 1114.8, 1010.3, 915.7, 1332.9, 1695.6, 1598.3, 2366.2, 2079.0, 1712.0, 1553.6, 1442.2, 1919.2, 1962.1, 1892.2, 1281.6, 863.4, 864.5, 1120.7, 739.0, 1231.4, 1323.4, 1016.8, 1438.0, 944.1, 1175.0, 1026.0, 666.5, 949.7, 1050.7, 958.1, 1043.2, 1079.0, 1257.9, 1248.1, 1094.1, 1370.2, 847.6, 926.4, 688.6, 1203.7, 1227.4, 949.4, 1121.6, 1467.0]
list2 = [939.5, 1046.7, 1074.2, 1161.6, 1054.3, 1132.8, 1046.9, 1048.2, 1253.0, 1048.5, 1014.0, 947.8, 1215.2, 1125.8, 1128.2, 771.9, 921.2, 903.2, 887.0, 995.7, 914.9, 1287.5, 902.7, 1318.9, 999.4, 1070.1, 818.2, 1111.9, 953.7, 786.1, 716.5, 1039.6, 1084.0, 784.7, 854.8, 772.3, 965.8, 748.8, 826.1, 1184.7, 847.8, 1107.3, 865.8, 1190.3, 1026.4, 1289.7, 1092.5, 953.4, 841.3, 937.0, 908.5, 1222.8, 1181.0, 941.9, 756.2, 656.2, 913.4, 797.9, 915.2, 1048.3, 1032.8, 884.0, 761.2, 855.8, 735.0, 703.0, 783.2, 1069.7, 888.4, 824.0, 869.2, 862.4, 769.3, 1079.7, 680.5, 837.1, 967.7, 739.3, 764.6, 972.7, 1384.5, 774.4, 804.3, 994.3, 1167.0, 747.3, 895.8, 1148.3, 1019.8, 792.4, 798.4, 774.6, 1030.2, 665.4, 619.0, 864.1, 819.1, 626.1, 726.0, 701.6, 888.8, 853.2, 721.3, 871.9, 960.7, 692.8, 617.5, 613.8, 597.1, 626.0, 596.6, 741.5, 505.3, 759.4, 452.3, 805.0, 683.3, 888.1, 828.0, 1110.8]
list3 = [1685.8, 873.5, 716.3, 353.4, 279.0, 51.9, 35.0, 81.1, 77.1, 36.0, 42.0, 52.8, 50.1, 44.5, 51.1, 37.6, 28.5, 33.7, 36.0, 43.2, 43.3, 51.0, 34.2, 49.1, 50.7, 50.5, 41.4, 39.2, 37.6, 29.2, 32.5, 51.9, 40.1, 36.9, 46.3, 40.1, 37.1, 40.1, 45.7, 43.1, 44.2, 51.8, 34.6, 47.7, 43.3, 52.5, 51.0, 46.2, 38.1, 37.3, 40.4, 49.2, 46.2, 32.7, 41.8, 27.2, 35.8, 31.3, 35.9, 41.2, 37.8, 47.5, 31.9, 28.8, 36.3, 31.8, 30.5, 37.7, 48.5, 35.1, 31.8, 38.1, 31.5, 27.0, 31.1, 28.2, 34.8, 25.9, 34.6, 51.3, 50.7, 51.1, 43.0, 45.1, 53.3, 38.5, 47.2, 40.4, 47.0, 41.4, 33.7, 39.1, 28.9, 38.0, 32.1, 36.0, 43.5, 35.3, 40.1, 29.5, 28.5, 29.8, 36.0, 37.6, 46.2, 32.4, 38.6, 29.4, 36.1, 30.5, 27.4, 33.7, 25.4, 36.1, 25.0, 46.3, 37.1, 36.8, 31.8, 51.5]

elist1 = [500.0, 500.0, 497.04, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
elist2 = [500.0, 497.87, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 498.38, 499.53, 499.7, 495.18, 493.48, 487.11, 485.17, 472.28, 488.74, 456.73, 465.3, 439.02, 430.77, 445.39, 417.01, 412.45, 409.55, 394.38, 409.69, 382.43, 374.44, 358.5, 350.89, 355.39, 342.19, 344.36, 323.46, 307.35, 316.92, 314.7, 300.36, 273.38, 274.17, 272.09, 276.13, 269.77, 246.66, 255.49, 244.16, 241.0, 218.91, 217.11, 223.44, 234.74, 196.61, 196.01, 202.46, 195.2, 187.15, 172.76, 169.16, 191.32, 149.06, 165.96, 156.66, 153.51, 143.54, 141.36, 146.18, 127.7, 138.93, 131.11, 128.16, 122.73, 123.15, 119.67, 114.86, 111.58, 112.2, 109.99, 113.15, 112.36, 110.2, 107.92, 103.52, 103.77, 95.18, 102.56, 96.23, 98.36, 94.7, 99.01, 101.13, 94.19, 93.98, 92.51, 94.49, 95.33, 96.11, 90.46, 89.32, 87.78, 87.82, 89.77, 87.38, 87.46, 88.44, 87.05, 83.97, 81.54, 86.46, 81.77, 81.7, 83.37, 83.01, 81.9, 80.94, 78.92, 77.1, 80.09, 78.87, 75.98, 78.16, 77.94, 77.38, 76.96, 78.23, 76.36, 75.24, 74.46, 76.62, 73.66, 76.78, 76.12, 73.09, 72.68, 74.92, 73.43, 72.96, 71.57, 72.94, 71.32, 70.7, 72.95, 70.92, 70.1, 70.29, 68.46, 69.62, 71.3, 70.2, 69.2, 68.18, 68.48, 69.03, 68.24, 66.44, 68.58, 68.88, 66.78, 65.0, 65.64, 66.84, 65.6, 67.97, 65.74, 65.92, 65.02, 65.12, 65.0, 65.4, 65.8, 64.42, 65.16, 64.54, 64.34, 63.46, 63.5, 63.8, 63.64, 62.38, 62.58, 62.14, 63.16, 61.58, 65.23, 61.3, 62.06, 61.1, 61.85, 60.88, 60.48, 61.3, 60.24, 60.58, 60.14, 59.72, 61.16, 60.64, 61.22, 61.02]
elist3 = [500.0, 500.0, 500.0, 500.0, 500.0, 497.86, 497.53, 497.58, 496.73, 494.84, 489.74, 484.19, 477.69, 473.58, 480.78, 473.46, 462.68, 457.62, 436.43, 431.41, 424.8, 409.01, 401.16, 402.77, 371.55, 368.4, 350.04, 343.53, 329.54, 309.52, 301.91, 320.27, 287.68, 273.18, 263.82, 252.68, 253.78, 225.01, 236.28, 234.82, 222.53, 217.67, 215.95, 212.74, 199.82, 187.83, 189.77, 184.45, 182.93, 176.43, 171.05, 166.69, 170.03, 159.8, 166.17, 157.07, 160.96, 154.8, 146.5, 147.59, 143.71, 145.38, 143.88, 137.43, 134.77, 127.85, 131.08, 124.69, 132.75, 127.18, 126.38, 123.46, 127.61, 118.43, 119.57, 122.98, 112.66, 115.22, 111.82, 112.42, 112.73, 111.06, 111.52, 107.75, 106.6, 107.39, 104.69, 104.27, 102.66, 102.82, 101.48, 100.94, 96.9, 101.25, 95.96, 96.89, 96.98, 96.89, 93.5, 92.24, 93.83, 94.02, 92.38, 92.01, 89.87, 90.4, 86.62, 89.04, 87.06, 86.68, 87.07, 87.06, 84.92, 86.88, 83.62, 85.71, 83.22, 82.99, 82.7, 84.03, 80.62, 78.46, 81.17, 81.27, 80.32, 79.9, 78.44, 80.0, 77.55, 78.93, 79.48, 77.08, 78.48, 76.8, 76.62, 76.6, 75.4, 74.81, 74.8, 73.58, 72.22, 74.12, 74.78, 72.28, 74.04, 71.86, 72.5, 72.56, 72.02, 71.18, 72.16, 71.42, 70.42, 70.92, 69.16, 69.9, 69.7, 70.18, 69.28, 69.68, 69.04, 68.62, 68.66, 68.04, 68.2, 67.64, 67.62, 68.42, 67.44, 67.36, 67.0, 66.74, 67.62, 66.56, 66.7, 66.12, 65.82, 65.98, 67.24, 65.74, 65.1, 65.86, 65.0, 64.98, 64.78, 65.12, 64.76, 65.12, 64.26, 64.16, 64.64, 63.2, 63.64, 64.1, 63.18, 63.42, 62.06, 61.86, 62.82, 63.8]
# Plotting
plt.plot(list1, label='LTL aug. Q-learning', marker='o', linestyle='-')
plt.plot(list2, label='LTL aug. Q-learning with reward shapping', marker='x', linestyle='--')
plt.plot(list3, label='MLRLM', marker='s', linestyle='-.')
plt.title('Multi-task setup')
plt.xlabel('Task Index')
plt.ylabel('Timesteps')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
