from utils import constant
from tracker import SimpleTracker

def main():
  print("BEGIN")
  print("BUILD SimpleTracker")
  tracker = SimpleTracker('video', constant.DATA_SOURCES)
  print("END BUILD")
  print("BEGIN RUN")
  tracker.run()
  print("END RUN")
  print("END")

if __name__ == '__main__':
  main()
