"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import { transcribeAudio } from "../lib/api";
import type { VoiceState } from "../lib/types";

interface VoiceRecorderResult {
  voiceState: VoiceState;
  elapsed: number;           // seconds since recording started
  errorMessage: string | null;
  toggleRecording: () => void;
  clearError: () => void;
}

export function useVoiceRecorder(
  onTranscribed: (text: string) => void,
): VoiceRecorderResult {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRafRef = useRef<number | null>(null);
  const peakRmsRef = useRef<number>(0);

  const SILENCE_RMS_THRESHOLD = 0.015;

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startTimer = () => {
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const clearError = useCallback(() => {
    setErrorMessage(null);
    if (voiceState === "error") setVoiceState("idle");
  }, [voiceState]);

  const toggleRecording = useCallback(async () => {
    // Check browser support
    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceState("error");
      setErrorMessage(
        "Microphone access is blocked. Use HTTPS or localhost, or enable insecure origins in browser flags.",
      );
      return;
    }

    // Stop recording
    if (voiceState === "recording") {
      mediaRecorderRef.current?.stop();
      setVoiceState("transcribing");
      stopTimer();
      return;
    }

    // Start recording
    try {
      setErrorMessage(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      peakRmsRef.current = 0;

      type AudioCtxCtor = typeof AudioContext;
      const AudioCtx: AudioCtxCtor | undefined =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: AudioCtxCtor })
          .webkitAudioContext;
      if (AudioCtx) {
        const ctx = new AudioCtx();
        audioCtxRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        const buf = new Float32Array(analyser.fftSize);
        const tick = () => {
          analyser.getFloatTimeDomainData(buf);
          let sum = 0;
          for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
          const rms = Math.sqrt(sum / buf.length);
          if (rms > peakRmsRef.current) peakRmsRef.current = rms;
          analyserRafRef.current = requestAnimationFrame(tick);
        };
        tick();
      }

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (analyserRafRef.current !== null) {
          cancelAnimationFrame(analyserRafRef.current);
          analyserRafRef.current = null;
        }
        if (audioCtxRef.current) {
          try {
            await audioCtxRef.current.close();
          } catch {}
          audioCtxRef.current = null;
        }

        if (peakRmsRef.current < SILENCE_RMS_THRESHOLD) {
          setVoiceState("error");
          setErrorMessage(
            "No speech detected. Move closer to the microphone and try again.",
          );
          return;
        }

        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setVoiceState("transcribing");

        const result = await transcribeAudio(blob);
        if (result.ok) {
          onTranscribed(result.text);
          setVoiceState("idle");
        } else {
          setVoiceState("error");
          setErrorMessage(result.error.message);
        }
      };

      mr.start();
      setVoiceState("recording");
      startTimer();
    } catch (err: unknown) {
      stopTimer();
      setVoiceState("error");
      const msg = err instanceof Error ? err.message : "Could not access microphone.";
      setErrorMessage(msg);
    }
  }, [voiceState, onTranscribed]);

  return { voiceState, elapsed, errorMessage, toggleRecording, clearError };
}
