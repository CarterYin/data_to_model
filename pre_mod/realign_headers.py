#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def realign_tsv_headers():
    """
    在第一列添加'samples'列名，原列名整体右移，使数据与列名对齐
    """
    input_file = "analysis_result_new.tsv"
    output_file = "analysis_result_new_realigned.tsv"
    
    try:
        print("🔄 开始处理文件...")
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # 处理第一行（列名行）
            first_line = infile.readline()
            if first_line:
                # 在原列名前添加 "samples\t"，原列名整体右移
                new_header = "samples\t" + first_line
                outfile.write(new_header)
                print("✅ 列名行已处理：添加了 'samples' 列名，原列名整体右移")
                
                # 显示处理效果
                original_cols = first_line.strip().split('\t')[:5]  # 前5个原列名
                new_cols = new_header.strip().split('\t')[:6]       # 前6个新列名
                
                print(f"📋 原始前5列名: {original_cols}")
                print(f"📋 新的前6列名: {new_cols}")
            
            # 复制所有数据行（保持不变）
            line_count = 0
            for line in infile:
                outfile.write(line)
                line_count += 1
                if line_count % 50000 == 0:  # 每处理50000行显示进度
                    print(f"📊 已处理 {line_count} 行数据...")
            
            print(f"✅ 所有数据行已复制完成，共 {line_count} 行")
        
        print(f"🎉 处理完成！")
        print(f"💾 输出文件：{output_file}")
        
        # 验证结果
        print(f"\n🔍 验证对齐效果：")
        with open(output_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            first_data = f.readline().strip()
        
        header_cols = header.split('\t')
        data_cols = first_data.split('\t')
        
        print(f"列名数量: {len(header_cols)}")
        print(f"数据列数量: {len(data_cols)}")
        
        if len(header_cols) == len(data_cols):
            print("✅ 列名与数据完美对齐！")
            print(f"\n📋 对齐效果预览：")
            for i in range(min(5, len(header_cols))):
                print(f"  '{header_cols[i]}' → '{data_cols[i]}'")
            if len(header_cols) > 5:
                print(f"  ... (还有 {len(header_cols) - 5} 列)")
        else:
            print("⚠️  列数不匹配，请检查")
            
        return True
        
    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return False

if __name__ == "__main__":
    realign_tsv_headers() 