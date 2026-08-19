from rineya_harness.web.extractor import ReadableTextExtractor


def test_extractor_removes_navigation_and_scripts() -> None:
    html = """
    <html><head><title>Test</title><script>evil()</script></head>
    <body><nav>Menu</nav><main><h1>Heading</h1><p>Useful text.</p></main><footer>Footer</footer></body>
    </html>
    """
    title, text = ReadableTextExtractor().extract(html)
    assert title == "Test"
    assert "Useful text." in text
    assert "Menu" not in text
    assert "evil" not in text
