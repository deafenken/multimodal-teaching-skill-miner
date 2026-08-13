export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  return Response.json(
    {
      status: "healthy",
      service: "teachlab-console",
      version: process.env.TEACHLAB_RELEASE_VERSION ?? "development",
      release_id: process.env.TEACHLAB_RELEASE_ID ?? "unsealed",
    },
    {status: 200, headers: {"Cache-Control": "no-store, max-age=0"}},
  );
}
