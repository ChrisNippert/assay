import threading


def parallel_map(func, items, workers=4):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an int >= 1")
    it = iter(items)
    lock = threading.Lock()
    results = {}
    errors = {}
    state = {"next": 0, "failed": False}

    def claim():
        with lock:
            if state["failed"]:
                return None
            try:
                item = next(it)
            except StopIteration:
                return None
            idx = state["next"]
            state["next"] += 1
            return idx, item

    def worker():
        while True:
            job = claim()
            if job is None:
                return
            idx, item = job
            try:
                value = func(item)
            except BaseException as e:
                with lock:
                    errors[idx] = e
                    state["failed"] = True
                return
            with lock:
                results[idx] = value

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[min(errors)]
    return [results[i] for i in range(len(results))]
