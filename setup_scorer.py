import os

if __name__ == "__main__":
    # NQG Scorer
    os.system(
        "sudo apt-get -y update && sudo apt-get -y install default-jre && sudo apt-get -y install default-jdk"
    )
    os.system("git clone -b python3 https://github.com/p208p2002/nqg.git")