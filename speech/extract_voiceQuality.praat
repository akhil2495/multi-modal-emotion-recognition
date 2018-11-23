form Test command line calls
    text file_name
endform

Read from file: file_name$

sound = selected ("Sound")
selectObject: sound
pitch = To Pitch: 0, 60, 600
selectObject: sound
pulses = To PointProcess (periodic, cc): 60, 600
selectObject: sound, pitch, pulses
voiceReport$ = Voice report: 0, 0, 60, 600, 1.3, 1.6, 0.03, 0.45

jitter_loc = extractNumber (voiceReport$, "Jitter (local): ")
jitter_loc_abs = extractNumber (voiceReport$, "Jitter (local, absolute): ")
jitter_rap = extractNumber (voiceReport$, "Jitter (rap): ")
jitter_ppq5 = extractNumber (voiceReport$, "Jitter (ddp): ")

shimmer_loc = extractNumber (voiceReport$, "Shimmer (local): ")
shimmer_loc_db = extractNumber (voiceReport$, "Shimmer (local, dB): ")
shimmer_apq3 = extractNumber (voiceReport$, "Shimmer (apq3): ")
shimmer_apq5 = extractNumber (voiceReport$, "Shimmer (apq5): ")
shimmer_apq11 = extractNumber (voiceReport$, "Shimmer (apq11): ")
shimmer_dda = extractNumber (voiceReport$, "Shimmer (dda): ")

unvoicedvoicedratio = extractNumber (voiceReport$, "Fraction of locally unvoiced frames: ")

meanautocorr = extractNumber (voiceReport$, "Mean autocorrelation: ")
mn2h = extractNumber (voiceReport$, "Mean noise-to-harmonics ratio: ")
mh2n = extractNumber (voiceReport$, "Mean harmonics-to-noise ratio: ")

writeFileLine: "temp.voiceQuality", jitter_loc, " ", jitter_loc_abs, " ", jitter_rap, " ", jitter_ppq5

appendFileLine: "temp.voiceQuality", shimmer_loc, " ", shimmer_loc_db, " ", shimmer_apq3, " ", shimmer_apq5, " ", shimmer_apq11, " ", shimmer_dda

appendFileLine: "temp.voiceQuality", unvoicedvoicedratio

appendFileLine: "temp.voiceQuality", meanautocorr, " ", mn2h, " ", mh2n
