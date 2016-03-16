from pytube import YouTube

yt = YouTube("https://www.youtube.com/watch?v=eZK2_-rIzJE")

# Once set, you can see all the codec and quality options YouTube has made
# available for the perticular video by printing videos.
print(yt.filename)
print yt.filter()

#downloads FLV from youtube
video = yt.get('flv')
video.download('data/{}.flv'.format(yt.filename.split()[0].lower()))
