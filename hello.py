




import smtplib
import getpass

from email.mime.text import MIMEText

def send_email():
    sender_address = 'manujg.it.21@nitj.ac.in'
    password = getpass.getpass(prompt='Enter your Gmail password: ')
    subject = "MANUJ_subject"
    message_body = '''
     hello guys
     i
     mean hello boys
     hii
     to all
     of you
     Thank you!
     MANUJ GUPTA
    '''
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_address, password)

    msg = MIMEText(message_body)
    msg['Subject'] = subject
    msg['From'] = sender_address
    msg['To'] = 'manujg.it.21@nitj.ac.in'
    recipients = 'manujg.it.21@nitj.ac.in'
    server.sendmail(sender_address, recipients, msg.as_string())

    server.quit()

send_email()

