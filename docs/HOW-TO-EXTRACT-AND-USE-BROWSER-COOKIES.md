# How to Extract and Use Browser Cookies for YouTube

YaClip uses **cookies** to authenticate with YouTube when downloading videos. This helps bypass
bot detection and allows access to member-only or age-restricted content.

We **do not** auto-extract cookies from your browser (the old mechanism was fragile and
platform-specific). Instead, you manually export your YouTube cookies once and provide the file
to YaClip.

## Step 1 — Install a Cookie Export Extension

Choose one for your browser:

| Browser | Extension |
|---|---|
| **Chrome / Brave / Edge** | [Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc?hl=en) |
| **Firefox** | [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) |

## Step 2 — Export YouTube Cookies

1. Go to [https://www.youtube.com](https://www.youtube.com) and **make sure you are logged in**.
2. Click the extension icon in your browser toolbar.
3. Click **Export** (the extension will download a `cookies.txt` file).
4. Save this file somewhere safe. The exported file uses the **Netscape cookie format** — this is the standard format that yt-dlp and YaClip accept.

> **Tip:** Cookies expire. If downloads start failing with 403 / "Sign in" errors, re-export
> a fresh `cookies.txt`.

## Step 3 — Use with YaClip

### CLI

```bash
python app.py clip <YouTube-URL> --cookies-file /path/to/cookies.txt
```

Or with the shorthand:

```bash
python app.py clip <YouTube-URL> -c cookies.txt
```

### WebUI

1. Launch the WebUI: `python app.py serve`
2. Open the **Clipper** tab.
3. Under the YouTube URL field, find the **Cookies File (Optional)** upload button.
4. Upload your `cookies.txt` file.
5. Configure the rest of your clip settings and click **Find Clips**.

YaClip will use the cookies for authentication while downloading.

## Without Cookies

If you don't provide a cookies file, YaClip will download videos without authentication.
This works for most public YouTube videos, but you may encounter:

- Rate limiting (429 errors) on bulk downloads
- Age-restricted content being blocked
- Member-only content unavailable

For normal use with public videos, cookies are **optional**.
