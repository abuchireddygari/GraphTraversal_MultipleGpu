from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib import colors
from glob import glob
from collections import defaultdict
from math import floor, ceil

cpu_timing_files = glob('*.timing')
gpu_timing_files = glob('*.timing_gpu')

data = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))

for filename in cpu_timing_files:
  graph, algorithm, foo = filename.split('.')
  with open(filename) as f:
    time = float(f.read())
  data[algorithm][graph]['time_cpu'] = time

for filename in gpu_timing_files:
  graph, algorithm, foo = filename.split('.')
  with open(filename) as f:
    time = float(f.read())
  data[algorithm][graph]['time_gpu'] = time

for algo in data:
  speedup = [[]]
  graphs = []
  for graph in data[algo]:
    cpu_time = data[algo][graph]['time_cpu']
    gpu_time = data[algo][graph]['time_gpu']
    speedup[0].append(cpu_time / gpu_time)
    graphs.append(graph)

  drawing = Drawing(400, 300)

  bc = VerticalBarChart()
  bc.x = 50
  bc.y = 50
  bc.height = 200
  bc.width = 300
  bc.data = speedup
  bc.strokeColor = colors.black

  bc.valueAxis.valueMin = 0
  bc.valueAxis.valueMax = max(speedup[0]) + 1
  bc.valueAxis.valueStep = max(1, int(floor(max(speedup[0]) / 5.)))
  bc.valueAxis.tickRight = 300
  bc.valueAxis.strokeDashArray = [2, 2]

  bc.categoryAxis.labels.boxAnchor = 'ne'
  bc.categoryAxis.labels.dx = 8
  bc.categoryAxis.labels.dy = -2
  bc.categoryAxis.labels.angle = 30
  bc.categoryAxis.categoryNames = graphs

  drawing.add(bc)
  drawing.save(fnRoot=algo + '_timing', formats=['pdf'])
