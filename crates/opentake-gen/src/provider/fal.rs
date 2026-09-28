//! fal.ai adapter (`prefix = "fal"`) — queue API. Submit returns a `request_id`;
//! polling reads `.../status` and, once terminal, fetches the output. Status
//! mapping: IN_QUEUE -> Queued, IN_PROGRESS -> Running, COMPLETED -> Succeeded,
//! FAILED -> Failed. See gen-SPEC §2.2.1.

use super::{normalize_output_urls, ModelRoute, ProviderAdapter};
use crate::error::{map_http_response, GenError};
use crate::job::{GenerationJob, JobStatus};
use crate::params::GenerationParams;
use crate::transport::{file_upload_body, HttpRequest, HttpTransport};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

const QUEUE_BASE: &str = "https://queue.fal.run";
const STORAGE_UPLOAD: &str = "https://rest.alpha.fal.ai/storage/upload";

#[derive(serde::Serialize, serde::Deserialize)]
struct FalJobRoutes {
    request_id: String,
    status_url: String,
    response_url: String,
}

/// fal.ai provider adapter.
pub struct FalAdapter {
    http: Arc<dyn HttpTransport>,
    api_key: String,
    queue_base: String,
}

impl FalAdapter {
    pub fn new(http: Arc<dyn HttpTransport>, api_key: impl Into<String>) -> Self {
        Self {
            http,
            api_key: api_key.into(),
            queue_base: QUEUE_BASE.to_string(),
        }
    }

    /// Test constructor with a no-op transport and dummy key. Used where the
    /// transport is irrelevant (e.g. registry routing).
    pub fn for_test() -> Self {
        Self::new(Arc::new(crate::transport::MockTransport::new()), "test-key")
    }

    /// Override the queue base (tests point this at a mock host).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.queue_base = base.into();
        self
    }

    fn auth_header(&self) -> (String, String) {
        ("Authorization".to_string(), format!("Key {}", self.api_key))
    }

    fn queue_url(&self, value: &str) -> Result<String, GenError> {
        let base = url::Url::parse(&self.queue_base)?;
        let candidate = url::Url::parse(value)?;
        if base.origin() != candidate.origin()
            || candidate.username() != ""
            || candidate.password().is_some()
        {
            return Err(GenError::Protocol(
                "fal returned a status URL outside its queue origin".into(),
            ));
        }
        Ok(candidate.to_string())
    }

    fn endpoint(route: &ModelRoute, params: &GenerationParams) -> String {
        match params {
            GenerationParams::Video(p)
                if route.vendor_model == "fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
                    && p.start_frame_url.is_some() =>
            {
                "fal-ai/kling-video/v2.5-turbo/pro/image-to-video".into()
            }
            GenerationParams::Image(p)
                if route.vendor_model == "fal-ai/flux-pro/kontext/text-to-image"
                    && !p.image_urls.is_empty() =>
            {
                "fal-ai/flux-pro/kontext".into()
            }
            _ => route.vendor_model.clone(),
        }
    }

    /// Map unified params into the fal request body (vendor field names).
    fn map_body(endpoint: &str, params: &GenerationParams) -> serde_json::Value {
        use serde_json::json;
        match params {
            GenerationParams::Image(p) => {
                let mut body = json!({ "prompt": p.prompt, "aspect_ratio": p.aspect_ratio });
                if let Some(r) = &p.resolution {
                    body["image_size"] = json!(r);
                }
                match p.image_urls.as_slice() {
                    [] => {}
                    [single] => body["image_url"] = json!(single),
                    many => body["image_urls"] = json!(many),
                }
                if p.num_images > 1 {
                    body["num_images"] = json!(p.num_images);
                }
                if endpoint == "fal-ai/flux-pro/v1.1" {
                    body.as_object_mut().unwrap().remove("aspect_ratio");
                    let image_size = p.resolution.as_deref().and_then(|size| {
                        let (w, h) = size.split_once('x')?;
                        Some(json!({ "width": w.parse::<u32>().ok()?, "height": h.parse::<u32>().ok()? }))
                    });
                    body["image_size"] = image_size.unwrap_or_else(|| {
                        json!(match p.aspect_ratio.as_str() {
                            "1:1" => "square_hd",
                            "9:16" => "portrait_16_9",
                            "3:4" => "portrait_4_3",
                            "16:9" => "landscape_16_9",
                            _ => "landscape_4_3",
                        })
                    });
                } else if endpoint.starts_with("fal-ai/flux-pro/kontext") {
                    body.as_object_mut().unwrap().remove("image_size");
                }
                body
            }
            GenerationParams::Video(p) => {
                let mut body = json!({
                    "prompt": p.prompt,
                    "duration": p.duration,
                    "aspect_ratio": p.aspect_ratio,
                    "enable_audio": p.generate_audio,
                });
                if let Some(r) = &p.resolution {
                    body["resolution"] = json!(r);
                }
                if let Some(u) = &p.start_frame_url {
                    body["image_url"] = json!(u);
                }
                if let Some(u) = &p.end_frame_url {
                    body["end_image_url"] = json!(u);
                }
                if !p.reference_image_urls.is_empty() {
                    body["reference_image_urls"] = json!(p.reference_image_urls);
                }
                if endpoint.starts_with("fal-ai/kling-video/") {
                    let map = body.as_object_mut().unwrap();
                    map.insert("duration".into(), json!(p.duration.to_string()));
                    map.remove("enable_audio");
                    map.remove("resolution");
                    map.remove("reference_image_urls");
                    if let Some(end) = map.remove("end_image_url") {
                        map.insert("tail_image_url".into(), end);
                    }
                }
                body
            }
            GenerationParams::Audio(p) => {
                let mut body = json!({ "prompt": p.prompt });
                if let Some(d) = p.duration_seconds {
                    body["duration"] = json!(d);
                }
                if let Some(u) = &p.video_url {
                    body["video_url"] = json!(u);
                }
                body
            }
            GenerationParams::Upscale(p) => {
                json!({ "video_url": p.source_url, "image_url": p.source_url })
            }
        }
    }

    /// Normalize a fal status string into our `JobStatus`.
    fn map_status(s: &str) -> JobStatus {
        match s {
            "IN_QUEUE" => JobStatus::Queued,
            "IN_PROGRESS" => JobStatus::Running,
            "COMPLETED" => JobStatus::Succeeded,
            _ => JobStatus::Failed,
        }
    }

    /// Extract result URLs from a fal `output`/result payload.
    fn extract_urls(output: &serde_json::Value) -> Vec<String> {
        for key in ["images", "video", "audio"] {
            if let Some(v) = output.get(key) {
                let urls = normalize_output_urls(v);
                if !urls.is_empty() {
                    return urls;
                }
            }
        }
        // Some models nest under "output".
        if let Some(inner) = output.get("output") {
            return Self::extract_urls(inner);
        }
        normalize_output_urls(output)
    }

    /// Read an accepted submission's routing and status.
    fn accepted_job(
        &self,
        endpoint: String,
        resp: &crate::transport::HttpResponse,
    ) -> Result<GenerationJob, GenError> {
        let v: serde_json::Value = resp.json()?;
        let request_id = v
            .get("request_id")
            .and_then(|x| x.as_str())
            .ok_or_else(|| GenError::Protocol("fal: missing request_id".into()))?;
        // Encode the routing needed for polling into the job id.
        let job_id = if let (Some(status_url), Some(response_url)) = (
            v.get("status_url").and_then(|value| value.as_str()),
            v.get("response_url").and_then(|value| value.as_str()),
        ) {
            serde_json::to_string(&FalJobRoutes {
                request_id: request_id.into(),
                status_url: self.queue_url(status_url)?,
                response_url: self.queue_url(response_url)?,
            })?
        } else {
            format!("{endpoint}|{request_id}")
        };
        let status = v
            .get("status")
            .and_then(|x| x.as_str())
            .map(Self::map_status)
            .unwrap_or(JobStatus::Queued);
        Ok(GenerationJob::pending(job_id, status))
    }
}

#[async_trait]
impl ProviderAdapter for FalAdapter {
    fn prefix(&self) -> &'static str {
        "fal"
    }

    async fn submit(
        &self,
        route: &ModelRoute,
        params: &GenerationParams,
    ) -> Result<GenerationJob, GenError> {
        let endpoint = Self::endpoint(route, params);
        let url = format!("{}/{}", self.queue_base, endpoint);
        let body = Self::map_body(&endpoint, params);
        let (hk, hv) = self.auth_header();
        let resp = self
            .http
            .send(HttpRequest::post(url).header(hk, hv).json(body))
            .await?;
        if !resp.is_success() {
            return Err(map_http_response(&resp));
        }
        // The provider accepted the request: anything unusable in its answer
        // leaves the submission's outcome unknown.
        self.accepted_job(endpoint, &resp)
            .map_err(GenError::after_success)
    }

    async fn poll(&self, job_id: &str) -> Result<GenerationJob, GenError> {
        let (status_url, result_url) = if job_id.starts_with('{') {
            let routes: FalJobRoutes = serde_json::from_str(job_id)?;
            (
                self.queue_url(&routes.status_url)?,
                self.queue_url(&routes.response_url)?,
            )
        } else {
            let (vendor_model, request_id) = job_id
                .split_once('|')
                .ok_or_else(|| GenError::Protocol("fal: malformed job id".into()))?;
            let app = vendor_model
                .split('/')
                .take(2)
                .collect::<Vec<_>>()
                .join("/");
            (
                format!("{}/{app}/requests/{request_id}/status", self.queue_base),
                format!("{}/{app}/requests/{request_id}", self.queue_base),
            )
        };
        let (hk, hv) = self.auth_header();
        let resp = self
            .http
            .send(HttpRequest::get(status_url).header(hk.clone(), hv.clone()))
            .await?;
        if !resp.is_success() {
            return Err(map_http_response(&resp));
        }
        let sv: serde_json::Value = resp.json()?;
        let status = sv
            .get("status")
            .and_then(|x| x.as_str())
            .map(Self::map_status)
            .unwrap_or(JobStatus::Failed);

        match status {
            JobStatus::Succeeded => {
                // Results can carry inline data URLs.
                let rresp = self
                    .http
                    .send(
                        HttpRequest::get(result_url)
                            .header(hk, hv)
                            .max_response_bytes(crate::transport::MEDIA_RESPONSE_BYTES_MAX),
                    )
                    .await?;
                if !rresp.is_success() {
                    return Err(map_http_response(&rresp));
                }
                let output: serde_json::Value = rresp.json()?;
                let urls = Self::extract_urls(&output);
                Ok(GenerationJob::succeeded(job_id, urls))
            }
            JobStatus::Failed => {
                let msg = sv
                    .get("error")
                    .and_then(|x| x.as_str())
                    .unwrap_or("Generation failed")
                    .to_string();
                Ok(GenerationJob::failed(job_id, msg))
            }
            other => Ok(GenerationJob::pending(job_id, other)),
        }
    }

    async fn upload(&self, path: &Path, content_type: &str) -> Result<String, GenError> {
        // fal storage upload: POST the file, receive a hosted URL. The file is
        // size-checked first and streamed from disk.
        let body = file_upload_body(path, content_type).await?;
        let (hk, hv) = self.auth_header();
        let resp = self
            .http
            .send(HttpRequest::post(STORAGE_UPLOAD).header(hk, hv).file(body))
            .await?;
        if !resp.is_success() {
            return Err(map_http_response(&resp));
        }
        let v: serde_json::Value = resp.json()?;
        v.get("access_url")
            .or_else(|| v.get("url"))
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| GenError::Protocol("fal: upload missing url".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ImageParams, VideoParams};
    use crate::transport::{Body, Method, MockTransport};
    use serde_json::json;

    fn adapter(mock: &MockTransport) -> FalAdapter {
        FalAdapter::new(Arc::new(mock.clone()), "fal-secret").with_base("https://mockfal")
    }

    #[tokio::test]
    async fn image_submit_maps_single_image_url_and_returns_job() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/flux-kontext",
            200,
            json!({"request_id": "req-1", "status": "IN_QUEUE"}),
        );
        let a = adapter(&mock);
        let route = ModelRoute::parse("fal:flux-kontext").unwrap();
        let params = GenerationParams::Image(ImageParams {
            prompt: "cat".into(),
            aspect_ratio: "1:1".into(),
            resolution: None,
            quality: None,
            image_urls: vec!["https://x/ref.png".into()],
            num_images: 1,
        });
        let job = a.submit(&route, &params).await.unwrap();
        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.id, "flux-kontext|req-1");

        // Verify the request body used image_url (single) not image_urls.
        let last = mock.last_call().unwrap();
        match last.body {
            Body::Json(v) => {
                assert_eq!(v["image_url"], "https://x/ref.png");
                assert!(v.get("image_urls").is_none());
            }
            _ => panic!("expected json"),
        }
        // Auth header present.
        assert!(last
            .headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Key fal-secret"));
    }

    #[tokio::test]
    async fn image_submit_maps_multiple_image_urls() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/flux-pro",
            200,
            json!({"request_id": "r", "status": "IN_QUEUE"}),
        );
        let a = adapter(&mock);
        let route = ModelRoute::parse("fal:flux-pro").unwrap();
        let params = GenerationParams::Image(ImageParams {
            prompt: "x".into(),
            aspect_ratio: "1:1".into(),
            resolution: None,
            quality: None,
            image_urls: vec!["a".into(), "b".into()],
            num_images: 2,
        });
        a.submit(&route, &params).await.unwrap();
        match mock.last_call().unwrap().body {
            Body::Json(v) => {
                assert_eq!(v["image_urls"], json!(["a", "b"]));
                assert_eq!(v["num_images"], 2);
            }
            _ => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn builtin_flux_route_uses_vendor_path_and_valid_image_size() {
        // https://fal.ai/models/fal-ai/flux-pro/v1.1/api
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            200,
            json!({"request_id": "r"}),
        );
        let a = adapter(&mock);
        let mut route = ModelRoute::parse("fal:flux-pro").unwrap();
        route.vendor_model = "fal-ai/flux-pro/v1.1".into();
        let mut params = ImageParams::new("cat", "16:9", 1);
        params.resolution = Some("1024x1536".into());
        a.submit(&route, &GenerationParams::Image(params))
            .await
            .unwrap();
        match mock.last_call().unwrap().body {
            Body::Json(body) => {
                assert_eq!(body["image_size"], json!({"width": 1024, "height": 1536}));
                assert!(body.get("aspect_ratio").is_none());
            }
            _ => panic!("expected JSON"),
        }
    }

    #[tokio::test]
    async fn builtin_kling_uses_image_endpoint_and_returned_poll_urls() {
        // https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/pro/image-to-video/api
        let mock = MockTransport::new();
        let endpoint = "https://mockfal/fal-ai/kling-video/v2.5-turbo/pro/image-to-video";
        let status = "https://mockfal/fal-ai/kling-video/requests/r/status";
        let response = "https://mockfal/fal-ai/kling-video/requests/r";
        mock.on(
            Method::Post,
            endpoint,
            200,
            json!({"request_id":"r", "status_url":status, "response_url":response}),
        );
        mock.on(Method::Get, status, 200, json!({"status":"COMPLETED"}));
        mock.on(
            Method::Get,
            response,
            200,
            json!({"video":{"url":"https://out/video.mp4"}}),
        );
        let a = adapter(&mock);
        let mut route = ModelRoute::parse("fal:kling-video").unwrap();
        route.vendor_model = "fal-ai/kling-video/v2.5-turbo/pro/text-to-video".into();
        let params = VideoParams {
            prompt: "scene".into(),
            duration: 5,
            aspect_ratio: "16:9".into(),
            start_frame_url: Some("https://x/start.png".into()),
            end_frame_url: Some("https://x/end.png".into()),
            ..Default::default()
        };
        let job = a
            .submit(&route, &GenerationParams::Video(params))
            .await
            .unwrap();
        let request = mock.last_call().unwrap();
        match request.body {
            Body::Json(body) => {
                assert_eq!(body["image_url"], "https://x/start.png");
                assert_eq!(body["tail_image_url"], "https://x/end.png");
                assert_eq!(body["duration"], "5");
                assert!(body.get("enable_audio").is_none());
                assert!(body.get("resolution").is_none());
            }
            _ => panic!("expected JSON"),
        }
        let done = a.poll(&job.id).await.unwrap();
        assert_eq!(done.result_urls, Some(vec!["https://out/video.mp4".into()]));
    }

    #[tokio::test]
    async fn video_submit_maps_frames_and_audio_flag() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/kling-video",
            200,
            json!({"request_id": "r"}),
        );
        let a = adapter(&mock);
        let route = ModelRoute::parse("fal:kling-video").unwrap();
        let params = GenerationParams::Video(VideoParams {
            prompt: "scene".into(),
            duration: 5,
            aspect_ratio: "16:9".into(),
            resolution: Some("1080p".into()),
            start_frame_url: Some("https://x/s.png".into()),
            end_frame_url: Some("https://x/e.png".into()),
            reference_image_urls: vec!["https://x/r.png".into()],
            generate_audio: false,
            ..Default::default()
        });
        a.submit(&route, &params).await.unwrap();
        match mock.last_call().unwrap().body {
            Body::Json(v) => {
                assert_eq!(v["image_url"], "https://x/s.png");
                assert_eq!(v["end_image_url"], "https://x/e.png");
                assert_eq!(v["reference_image_urls"], json!(["https://x/r.png"]));
                assert_eq!(v["enable_audio"], false);
                assert_eq!(v["duration"], 5);
                assert_eq!(v["resolution"], "1080p");
            }
            _ => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn poll_queued_then_running_then_succeeded_with_output() {
        let mock = MockTransport::new();
        mock.on_sequence(
            Method::Get,
            "https://mockfal/kling-video/requests/req-9/status",
            vec![
                (200, json!({"status": "IN_QUEUE"})),
                (200, json!({"status": "IN_PROGRESS"})),
                (200, json!({"status": "COMPLETED"})),
            ],
        );
        mock.on(
            Method::Get,
            "https://mockfal/kling-video/requests/req-9",
            200,
            json!({"video": {"url": "https://out/v.mp4"}}),
        );
        let a = adapter(&mock);
        let id = "kling-video|req-9";
        assert_eq!(a.poll(id).await.unwrap().status, JobStatus::Queued);
        assert_eq!(a.poll(id).await.unwrap().status, JobStatus::Running);
        let done = a.poll(id).await.unwrap();
        assert_eq!(done.status, JobStatus::Succeeded);
        assert_eq!(done.result_urls, Some(vec!["https://out/v.mp4".into()]));
    }

    #[tokio::test]
    async fn poll_extracts_images_array() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://mockfal/flux-pro/requests/r/status",
            200,
            json!({"status": "COMPLETED"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/flux-pro/requests/r",
            200,
            json!({"images": [{"url": "https://out/a.png"}, {"url": "https://out/b.png"}]}),
        );
        let a = adapter(&mock);
        let done = a.poll("flux-pro|r").await.unwrap();
        assert_eq!(
            done.result_urls,
            Some(vec!["https://out/a.png".into(), "https://out/b.png".into()])
        );
    }

    #[tokio::test]
    async fn poll_failed_carries_error_message() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://mockfal/x/requests/r/status",
            200,
            json!({"status": "FAILED", "error": "nsfw blocked"}),
        );
        let a = adapter(&mock);
        let done = a.poll("x|r").await.unwrap();
        assert_eq!(done.status, JobStatus::Failed);
        assert_eq!(done.error_message.as_deref(), Some("nsfw blocked"));
    }

    #[tokio::test]
    async fn submit_http_error_maps_to_gen_error() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/flux-pro",
            401,
            json!({"error": {"code": "unauthenticated", "message": "bad key"}}),
        );
        let a = adapter(&mock);
        let route = ModelRoute::parse("fal:flux-pro").unwrap();
        let params = GenerationParams::Image(ImageParams::new("x", "1:1", 1));
        assert!(matches!(
            a.submit(&route, &params).await,
            Err(GenError::Unauthenticated)
        ));
    }

    #[tokio::test]
    async fn an_accepted_submission_without_usable_routing_has_an_unknown_outcome() {
        for body in [
            json!({"status": "IN_QUEUE"}),
            json!({
                "request_id": "r",
                "status_url": "https://elsewhere.test/status",
                "response_url": "https://elsewhere.test/result"
            }),
        ] {
            let mock = MockTransport::new();
            mock.on(Method::Post, "https://mockfal/flux-pro", 200, body.clone());
            let route = ModelRoute::parse("fal:flux-pro").unwrap();
            let params = GenerationParams::Image(ImageParams::new("x", "1:1", 1));
            let error = adapter(&mock).submit(&route, &params).await.unwrap_err();
            assert!(
                matches!(error, GenError::UnusableSuccess(_)),
                "{body}: {error:?}"
            );
            assert!(error.submission_outcome_unknown());
        }
    }

    #[test]
    fn status_mapping_is_exhaustive() {
        assert_eq!(FalAdapter::map_status("IN_QUEUE"), JobStatus::Queued);
        assert_eq!(FalAdapter::map_status("IN_PROGRESS"), JobStatus::Running);
        assert_eq!(FalAdapter::map_status("COMPLETED"), JobStatus::Succeeded);
        assert_eq!(FalAdapter::map_status("ANYTHING_ELSE"), JobStatus::Failed);
    }
}
