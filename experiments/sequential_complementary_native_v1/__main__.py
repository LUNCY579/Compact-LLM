"""Review entry point; native commands cannot consume a permit on this commit."""
import sys
from .boundary import NativeReviewLocked, main, require_native_review

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"qualify", "_worker", "propose"}:
        try:
            require_native_review()
        except NativeReviewLocked as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2)
    raise SystemExit(main())
