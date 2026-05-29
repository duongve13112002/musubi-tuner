> 📝 Click on the language section to expand / 言語をクリックして展開

# Multi-Caption Support

## Overview / 概要

Musubi Tuner supports multiple captions per training item. When `enable_multi_caption = true` is set in the dataset config, one caption is selected at random at each training step. This encourages the model to associate multiple descriptions with the same visual content.

Multi-caption is **disabled by default**. Set `enable_multi_caption = true` in your dataset config to enable it.

<details>
<summary>日本語</summary>

Musubi Tunerは、学習アイテムごとに複数のキャプションをサポートしています。データセット設定に`enable_multi_caption = true`を設定すると、各学習ステップでキャプションがランダムに選択されます。これにより、同じ視覚コンテンツに複数の説明を関連付けることができます。

マルチキャプションは**デフォルトで無効**です。有効にするにはデータセット設定に`enable_multi_caption = true`を設定してください。

</details>

## Enabling Multi-Caption / マルチキャプションの有効化

Add `enable_multi_caption = true` to your dataset config TOML:

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "/path/to/images"
cache_directory = "/path/to/cache"
enable_multi_caption = true
```

This can be set at the `[general]` level (applies to all datasets) or at the `[[datasets]]` level (per-dataset).

<details>
<summary>日本語</summary>

データセット設定TOMLに`enable_multi_caption = true`を追加してください。

`[general]`レベル（全データセットに適用）または`[[datasets]]`レベル（データセットごと）で設定できます。

</details>

## Caption File Formats / キャプションファイルの形式

### Multi-line `.txt` files / 複数行の`.txt`ファイル

Each non-empty line in a `.txt` caption file is treated as a separate caption. A file with a single line behaves as before.

```
# image1.txt — single caption (unchanged behavior)
A photo of a cat sitting on a windowsill

# image2.txt — three alternative captions (requires enable_multi_caption = true to rotate)
A cat on a windowsill
Tabby cat resting near a sunny window
Domestic cat overlooking the street
```

<details>
<summary>日本語</summary>

`.txt`キャプションファイルの空でない各行が個別のキャプションとして扱われます。1行のみのファイルは従来通りの動作です。

</details>

### JSONL files / JSONLファイル

Use the `captions` key with a list of strings instead of the single `caption` key:

```json
{"image_path": "/path/to/image1.jpg", "captions": ["First caption", "Second caption", "Third caption"]}
{"image_path": "/path/to/image2.jpg", "caption": "Old single-caption format still works"}
```

Both `caption` (single string) and `captions` (list) can be mixed in the same file.

<details>
<summary>日本語</summary>

単一の`caption`キーの代わりに、文字列のリストで`captions`キーを使用してください。

`caption`（単一文字列）と`captions`（リスト）は同じファイル内で混在させることができます。

</details>

## Caching Behavior / キャッシングの動作

The cache scripts always write **all** captions found in the source file. The `enable_multi_caption` flag only affects the **reading** side during training — it does not change what is cached.

- When `enable_multi_caption = false` (default): only `caption_0` is used during training, even if multiple captions are cached.
- When `enable_multi_caption = true`: a random caption is picked at each training step.

This means you can cache once and switch between single/multi-caption behavior by changing the TOML config, without re-caching.

<details>
<summary>日本語</summary>

キャッシュスクリプトは、ソースファイルで見つかった**すべての**キャプションを常に書き込みます。`enable_multi_caption`フラグは学習中の**読み取り**側にのみ影響します。キャッシュ内容は変わりません。

- `enable_multi_caption = false`（デフォルト）の場合：複数のキャプションがキャッシュされていても、学習中は`caption_0`のみが使用されます。
- `enable_multi_caption = true`の場合：各学習ステップでランダムにキャプションが選択されます。

つまり、一度キャッシュしておけば、再キャッシュせずにTOML設定を変更するだけでシングル/マルチキャプションの動作を切り替えることができます。

</details>

## Interaction with Caption Dropout / キャプションドロップアウトとの関係

Caption dropout (`caption_dropout_rate`) and multi-caption selection are applied in this order:

1. If `enable_multi_caption = true`: pick a random caption index from the cache.
2. Apply caption dropout: with probability `caption_dropout_rate`, replace the selected caption with the global empty embedding.

Caption dropout works regardless of whether `enable_multi_caption` is enabled.

<details>
<summary>日本語</summary>

キャプションドロップアウト（`caption_dropout_rate`）とマルチキャプション選択は以下の順序で適用されます。

1. `enable_multi_caption = true`の場合：キャッシュからランダムなキャプションインデックスを選択します。
2. キャプションドロップアウトを適用：`caption_dropout_rate`の確率で、選択されたキャプションをグローバルな空の埋め込みに置き換えます。

キャプションドロップアウトは`enable_multi_caption`の有効/無効に関わらず機能します。

</details>

## Backward Compatibility / 後方互換性

- Cache files written in the old single-caption format (without `caption_0_` prefixed keys) are automatically detected and used as-is, regardless of the `enable_multi_caption` setting.
- Existing single-line `.txt` files require no changes.
- Setting `enable_multi_caption = true` on a dataset with only single-caption cache files is safe — the single caption is always used.

<details>
<summary>日本語</summary>

- 旧来のシングルキャプション形式（`caption_0_`プレフィックスなし）で書かれたキャッシュファイルは自動的に検出され、`enable_multi_caption`設定に関係なくそのまま使用されます。
- 既存の1行の`.txt`ファイルに変更は必要ありません。
- シングルキャプションのキャッシュファイルのみのデータセットで`enable_multi_caption = true`を設定しても安全です。常に単一キャプションが使用されます。

</details>
