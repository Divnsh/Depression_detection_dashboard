import os

def deploy_tfserving():
    # tensorflow serving deploy
    os.system('bash servemodel.sh')
    print("Model deployed using tensorflow-serving.")

if __name__=='__main__':
    deploy_tfserving()
