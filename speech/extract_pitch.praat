form Test command line calls
    text file_name
endform

Read from file: file_name$

sound = selected ("Sound")
tmin = Get start time
tmax = Get end time
To Pitch: 0.005, 75, 300
Rename: "pitch"
selectObject: sound
writeFileLine: "temp.pitch", "Pitch 0.01"
for i to (tmax-tmin)/0.05
    time = tmin + i * 0.05
    selectObject: "Pitch pitch"
    pitch = Get value at time: time, "Hertz", "Linear"
    appendFileLine: "temp.pitch", fixed$ (time, 2), " ", fixed$ (pitch, 3)
endfor
