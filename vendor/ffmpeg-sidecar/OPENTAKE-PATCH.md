# Retained input support

Source: ffmpeg-sidecar 2.5.2 from crates.io, MIT license retained.

The only source change removes the piped-stdin assertion in
`FfmpegChild::from_inner`. stdout and stderr remain piped and validated.
FFmpeg can consume a seekable regular file through stdin using `fd:`;
requiring a pipe prevented Windows native filenames from reaching FFmpeg
without lossy UTF-16 conversion. `take_stdin` already returns `Option`,
and interactive commands already return an error when stdin is absent.

Remove this patch when upstream supports inherited file stdin.
The media integration tests exercise frame iteration, PCM, cancellation,
and a filename whose lossy spelling names a different file.
