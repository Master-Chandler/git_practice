import paramiko
ssh = paramiko.SSHClient()

# 自动认证
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('159.226.113.51',port='22',username='dam',password='IAP+100029')
stdin, stdout, stderr = ssh.exec_command('ls')
print(stdin.readlines(), stdout.readlines(), stderr.readlines())
ssh.close()
