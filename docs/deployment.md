# Deployment

The frontend is a Vite static site for Vercel. The live diagnosis API is a
Dockerized FastAPI service for Google Cloud Run. The proof library remains
available as static JSON even when the API is asleep or unavailable.

## Backend: Cloud Run

Cloud Run's request-based free allowance currently includes monthly CPU,
memory, and request quotas. It still requires a Google Cloud billing account,
and usage beyond the allowance can incur charges. Set a budget alert before
deploying and keep the maximum instance count at one for this demo.

1. In Google Cloud Console, create/select a project, enable Cloud Run and
   Artifact Registry, and configure a billing budget alert.
2. Create a Cloud Run service from this repository using its `Dockerfile`.
   Set the source directory to the repository root and the region nearest the
   expected users.
3. Set `CORS_ALLOW_ORIGINS` to the exact Vercel production origin, for example
   `https://model-doctor.vercel.app`.
4. Configure the service for request-based billing, 2 GiB memory, 0.5 CPU,
   concurrency `1`, and maximum instances `1`. Allow unauthenticated requests
   so the public frontend can call the API.
5. Deploy and verify `https://<service-url>/api/health` returns JSON with
   `status: ok`.

The image contains the trained doctor and downloads/preprocesses MNIST while
building, so live requests do not depend on an external dataset download.
Model checkpoint uploads are evaluated in memory/temp storage and are not
retained. The service may scale to zero between visits, so the first live
diagnosis can take longer.

## Frontend: Vercel

1. Import `Andrina-1409/MODEL-DOCTOR` into Vercel.
2. Set the project Root Directory to `frontend` and keep the Vite framework
   preset, build command `npm run build`, and output directory `dist`.
3. Add the environment variable `VITE_API_BASE_URL` with the Cloud Run service
   origin only, without `/api` at the end, for example
   `https://model-doctor-api-xxxxx.a.run.app`.
4. Deploy the production branch. Vercel builds this variable into the frontend,
   so redeploy after changing it.
5. Copy the final Vercel production origin into Cloud Run's
   `CORS_ALLOW_ORIGINS` and redeploy the backend if the origin changed.

The Vite development proxy keeps local development working when
`VITE_API_BASE_URL` is empty. Static proof/demo requests stay on the Vercel
origin and do not use the backend.

## Cost and availability notes

- Vercel Hobby is free for personal, non-commercial projects, subject to its
  current usage limits.
- Cloud Run is usage-based with a monthly free allowance, not an unconditional
  free always-on server. The service can sleep at zero traffic and cold-start.
- Do not enable min instances or use paid CPU/memory tiers for this demo unless
  you intend to pay for them.
