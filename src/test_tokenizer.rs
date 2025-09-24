use tokenizers::Tokenizer;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer_path = "./tokenizer.json";

    // 使用 ? 来处理错误
    let tokenizer = Arc::new(Tokenizer::from_file(tokenizer_path)?);

    let text = "你好，世界！这是一次分词测试。";
    println!("文本: {}", text);

    // 编码
    let encoding = tokenizer.encode(text, true)?;

    let token_ids = encoding.get_ids();
    let token_strs = encoding.get_tokens();

    println!("Token ID: {:?}", token_ids);

    println!("Token 字符串:");
    for (id, token) in token_ids.iter().zip(token_strs.iter()) {
        // decode 传切片引用
        let token_text = tokenizer.decode(&[*id], true)?;
        println!("ID: {:<6} Token: {}", id, token_text);
    }

    // 整体 decode
    let decoded_text = tokenizer.decode(token_ids, true)?;
    println!("整体 decode: {}", decoded_text);

    Ok(())
}
